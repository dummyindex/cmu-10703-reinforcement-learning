from hashlib import new
import sys
import argparse
import numpy as np

import torch


class A2C(object):
    # Implementation of N-step Advantage Actor Critic.

    def __init__(self, actor, actor_lr, N, nA, critic, critic_lr, baseline=False, a2c=True, type="Reinforce"):
        # Note: baseline is true if we use reinforce with baseline
        #       a2c is true if we use a2c else reinforce
        # TODO: Initializes A2C.
        self.type = type  # Pick one of: "A2C", "Baseline", "Reinforce"
        assert self.type is not None
        print("Training with {}.".format(self.type))
        self.actor = actor
        self.actor_lr = actor_lr
        self.N = N
        self.nA = nA
        self.critic = critic
        self.critic_lr = critic_lr
        self.baseline = baseline
        self.a2c = a2c
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr)
        
        if self.type == "Baseline" or self.type == "A2C":
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=self.critic_lr)

    def evaluate_policy(self, env):
        # TODO: Compute Accumulative trajectory reward(set a trajectory length threshold if you want)
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = self.actor(torch.from_numpy(state).float())
            # action = torch.argmax(action).item()
            action = torch.distributions.Categorical(action).sample().item()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        return episode_reward

    def evaluate_policy_episodes(self, env, num_episodes=1):
        all_rewards = []
        for i in range(num_episodes):
            all_rewards.append(self.evaluate_policy(env))
        return np.array(all_rewards)

    def generate_episode(self, env, render=False):
        from torch.distributions import Categorical
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        self.actor.eval()
        env.reset()
        cur_state = env.state
        states = []
        actions = []
        rewards = []
        t = 0

        while True:
            action_probs = self.actor(torch.from_numpy(cur_state).float())
            # action = torch.argmax(action).item()
            # action = np.random.choice(self.nA, p=action_probs.detach().numpy())
            action = Categorical(action_probs).sample().item()
            states.append(cur_state)
            actions.append(action)

            new_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
            cur_state = new_state
            t += 1
        return states, actions, rewards

    def train(self, env, gamma=0.99, n=10, num_episodes=1000):
        # Trains the model on a single episode using REINFORCE or A2C/A3C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        self.actor.train()
        self.actor_optimizer.zero_grad()
        states, actions, rewards = self.generate_episode(env)
        T = len(states)
        Gs = [] # returns
        G = 0
        for t in range(T-1, -1, -1):
            if self.type == "A2C":
                V_end = self.critic(torch.from_numpy(states[t + self.N]).float()) if t + self.N < T else 0
                G = rewards[t] + gamma * V_end
                t_end = min(t + self.N, T)
                for k in range(t, t_end):
                    G += gamma ** (k-t) * rewards[k]
            else:
                G = gamma * G + rewards[t]
            Gs.append(G)
        Gs.reverse()
        Gs = torch.tensor(Gs, requires_grad=True)

        state_tensors = [torch.tensor(
                state, requires_grad=True).float() for state in states]
        log_probs = torch.stack([torch.log(self.actor(state_tensor)[action]) for state_tensor, action in zip(state_tensors, actions)])
        
        # update actor/critic parameters
        if self.type == "Reinforce":      
            loss_per_t = [log_prob * G for log_prob, G in zip(log_probs, Gs)]
            assert loss_per_t.shape[0] == T
            loss_theta = - loss_per_t.sum() / T
            # print("loss_theta type: ", type(loss_theta))
            # print("loss_theta: {}".format(loss_theta))
            (loss_theta).backward()
            self.actor_optimizer.step()
        elif self.type == "Baseline" or self.type == "A2C":
            # update policy net
            critic_returns = torch.tensor([self.critic(torch.tensor(state).float()) for state in states], requires_grad=True)
            loss_per_t = (Gs - critic_returns) * log_probs
            assert loss_per_t.shape[0] == T
            loss_theta = - loss_per_t.sum() / T
            (loss_theta).backward() 
            self.actor_optimizer.step()

            # update critic baseline
            self.critic_optimizer.zero_grad()
            loss_w = (Gs - critic_returns).pow(2).sum() / T
            loss_w.backward()
            self.critic_optimizer.step()
        else:
            raise NotImplementedError("not implemented yet.")
