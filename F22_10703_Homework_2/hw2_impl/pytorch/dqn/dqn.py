#!/usr/bin/env python
from queue import Full
import numpy as np, gym, sys, copy, argparse
import os
import torch
import collections
import tqdm
import matplotlib.pyplot as plt

class FullyConnectedModel(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, 16)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 16)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(16, 16)
        self.activation3 = torch.nn.ReLU()

        self.output_layer = torch.nn.Linear(16, output_size)
        #no activation output layer

        #initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.output_layer(x)
        return x


QNetwork = ...
class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env, lr, logdir=None, gamma=0.99):
    # Define your network architecture here. It is also a good idea to define any training operations
    # and optimizers here, initialize your variables, or alternately compile your model here.
        self.net = FullyConnectedModel(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma

    def save_model_weights(self, suffix, model_file=None):
        # Helper function to save your model / weights.
        path = os.path.join(self.logdir, "model")
        if model_file is None:
            model_file = os.path.join(path, "model_{}.pth".format(suffix))
        torch.save(self.net.state_dict(), model_file)
        return path

    def load_model(self, model: QNetwork):
        """Helper function to copy model parameters."""
        self.net.load_state_dict(model.net.state_dict())

    def load_model_weights(self,weight_file):
        # Optional Helper function to load model weights.
        self.net.load_state_dict(torch.load(weight_file))

    def predict(self, state):
        """Helper function to predict Q values of actions for a given state."""
        self.net.eval()
        state = torch.from_numpy(state).float()
        return self.net(state)

    def train(self, batch_x, Q_target: QNetwork):
        """Helper function to train your model on a given batch of experience replay samples."""
        self.net.train()
        self.optimizer.zero_grad()
        
        ys = []
        qvalues_self = []
        for sample in batch_x:
            state, action, reward, next_state, done = sample
            state = torch.from_numpy(state).float()
            next_state = torch.from_numpy(next_state).float()
            action = torch.from_numpy(action).float()
            reward = torch.from_numpy(reward).float()

            y = None
            if done:
                y = reward
            else:
                y = reward + self.gamma * torch.max(Q_target.predict(next_state))
            ys.append(y)
            qvalues_self.append(self.predict(state)[action])
        
        loss = torch.nn.functional.mse_loss(qvalues_self, ys, reduction="mean")
        loss.backward()
        self.optimizer.step()
        

class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        # Hint: you might find this useful:
        # 		collections.deque(maxlen=memory_size)
        self.burn_in = burn_in
        self.memory_size = memory_size
        self.memory = collections.deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        import random
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        return random.sample(self.memory, batch_size)

    def append(self, transition):
        # Appends transition to the memory.
        self.memory.append(transition)
        if len(self.memory) > self.memory_size:
            self.memory.popleft()


class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, render=False, qw_lr=5e-4, epsilon=0.05):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.env = gym.make(environment_name)
        self.logdir = os.path.join(os.getcwd(), "agent_logs")
        self.Qw = QNetwork(self.env, lr=qw_lr, logdir=self.logdir)
        self.Q_target = QNetwork(self.env, lr=None, logdir=self.logdir)
        self.memory = Replay_Memory()
        self.epsilon = epsilon

        self.w_target_interval = 50
    

    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(q_values)

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values)

    def train_single_episode(self, max_episode_T=None):
        if max_episode_T is None:
            max_episode_T = self.env.spec.max_episode_steps
        cur_state = self.env.reset()
        for t in range(max_episode_T):
            # Sample action from epsilon greedy policy
            qvalues = self.Qw.predict(self.env.state)
            action = self.epsilon_greedy_policy(qvalues)
            # Take step in environment
            next_state, r, done, info = self.env.step(action)
            # Append transition to memory
            new_memory_entry = (cur_state, action, r, next_state, done)
            self.memory.append(new_memory_entry)
            # Sample batch from memory
            batch = self.memory.sample_batch()
            # Train network on batch
            self.Qw.train(batch)
            # Update target network
            if t % self.w_target_interval == 0:
                self.Q_target.load_model(self.Qw)
            cur_state = next_state


    def train(self, num_episodes=1000, max_episode_T=None):
        # In this function, we will train our network.

        # When use replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        for i in range(num_episodes):
            self.train_single_episode(max_episode_T)
            if i % 50 == 0:
                self.Qw.save_model_weights("episode_{}".format(i))

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 20 episodes, by calculating average cumulative rewards (returns) for the 20 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using replay memory.
        pass

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        self.env.reset()
        for i in range(self.memory.burn_in):
            cur_state = self.env.state
            action = self.env.action_space.sample()
            next_state, r, done, info = self.env.step(action)
            new_memory_entry = (cur_state, action, r, next_state)
            self.memory.append(new_memory_entry)

# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop.
def test_video(agent, env, epi):
    # Usage:
    # 	you can pass the arguments within agent.train() as:
    # 		if episode % int(self.num_episodes/3) == 0:
    #       	test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.epsilon_greedy_policy(state, 0.05)
        next_state, reward, done, info = env.step(action)
        state = next_state
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    agent.env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str,default='CartPole-v0')
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    parser.add_argument('--lr', dest='lr', type=float, default=5e-4)
    return parser.parse_args()


def main(args):

    args = parse_arguments()
    environment_name = args.env

    # You want to create an instance of the DQN_Agent class here, and then train / test it.

if __name__ == '__main__':
    main(sys.argv)
