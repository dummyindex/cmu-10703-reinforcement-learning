import torch
import tqdm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from net import NeuralNet
from a2c import A2C
import gym
import sys
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'


matplotlib.use('Agg')


def parse_a2c_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', dest='env_name', type=str,
                        default='CartPole-v0', help="Name of the environment to be run.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=3500, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--baseline-lr', dest='baseline_lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=100, help="The value of N in N-step A2C.")
    parser.add_argument('--net-type', dest='net_type', type=str,
                        default='Reinforce', help="network type: A2C, Baseline, or Reinforce")

    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main_a2c(args):
    # Parse command-line arguments.
    args = parse_a2c_arguments()
    env_name = args.env_name
    net_type = args.net_type

    num_episodes = args.num_episodes
    lr = args.lr
    baseline_lr = args.baseline_lr
    critic_lr = args.critic_lr
    # render = args.render

    # Create the environment.
    env = gym.make(env_name)
    nA = env.action_space.n
    nS = env.observation_space.shape[0]
    print("nA: {}, nS: {}".format(nA, nS))
    # Plot average performance of 5 trials
    num_seeds = 5
    l = num_episodes//100
    res = np.zeros((num_seeds, l))

    gamma = 0.99

    out_activation = torch.nn.functional.softmax
    for i in tqdm.tqdm(range(num_seeds)):
        reward_means = []

        # TODO: create networks and setup reinforce/a2c
        # TODO: some of the args are unnecessary for reinforce, but it's fine to leave them in
        if net_type == 'Reinforce':
            policy_net = NeuralNet(nS, nA, out_activation)
            A2C_net = A2C(policy_net, lr, args.n,
                      nA, None, critic_lr, baseline=False, a2c=True, type=net_type)
        elif net_type == 'Baseline' or net_type == 'A2C':
            policy_net = NeuralNet(nS, nA, out_activation)
            critic_net = NeuralNet(nS, 1, torch.nn.Identity(1, 1))
            A2C_net = A2C(policy_net, lr, args.n,
                      nA, critic_net, critic_lr, baseline=True, a2c=True, type=net_type)
        else:
            raise ValueError("net_type must be one of 'Reinforce', 'Baseline', or 'A2C'")

        for m in range(num_episodes):
            A2C_net.train(env, gamma=gamma)
            if m % 100 == 0:
                print("Episode: {}".format(m))
                G = np.zeros(20)
                for k in range(20):
                    g = A2C_net.evaluate_policy(env)
                    G[k] = g

                reward_mean = G.mean()
                reward_sd = G.std()
                print("The test reward for episode {0} is {1} with sd of {2}.".format(
                    m, reward_mean, reward_sd))
                reward_means.append(reward_mean)
        res[i] = np.array(reward_means)

    ks = np.arange(l)*100
    avs = np.mean(res, axis=0)
    maxs = np.max(res, axis=0)
    mins = np.min(res, axis=0)

    plt.fill_between(ks, mins, maxs, alpha=0.1)
    plt.plot(ks, avs, '-o', markersize=1)

    plt.xlabel('Episode', fontsize=15)
    plt.ylabel('Return', fontsize=15)

    if not os.path.exists('./plots'):
        os.mkdir('./plots')

    if A2C_net.type == 'A2C':
        plt.title("A2C Learning Curve for N = {}".format(args.n), fontsize=24)
        plt.savefig("./plots/a2c_curve_N={}.png".format(args.n))
    elif A2C_net.type == 'Baseline':
        plt.title("Baseline Reinforce Learning Curve".format(
            args.n), fontsize=24)
        plt.savefig("./plots/Baseline_Reinforce_curve.png".format(args.n))
    else:  # Reinforce
        plt.title("Reinforce Learning Curve", fontsize=24)
        plt.savefig("./plots/Reinforce_curve.png")


if __name__ == '__main__':
    main_a2c(sys.argv)
