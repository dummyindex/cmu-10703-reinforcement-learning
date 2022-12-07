import collections
from typing import Optional
from mcts import visit_softmax_temperature
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

MAX_FLOAT_VAL = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class TrainResults(object):

    def __init__(self, save_dir=None):
        self.value_losses = []
        self.reward_losses = []
        self.policy_losses = []
        self.total_losses = []
        self.save_dir = Path(save_dir)

    def plot_total_loss(self):
        x_vals = np.arange(len(self.total_losses))
        plt.plot(x_vals, self.total_losses, label="Train Loss")
        plt.xlabel("Train Steps")
        plt.ylabel("Loss")
        plt.savefig(self.save_dir / "total_loss.png")
        plt.show()

    def plot_individual_losses(self):
        x_vals = np.arange(len(self.total_losses))
        plt.plot(x_vals, self.value_losses, label="Value Loss")
        plt.plot(x_vals, self.policy_losses, label="Policy Loss")
        plt.plot(x_vals, self.reward_losses, label="Reward Loss")
        plt.xlabel("Train Steps")
        plt.ylabel("Losses")
        plt.legend()
        plt.savefig(self.save_dir / "individual_losses.png")
        plt.show()

    def plot_policy_loss(self):
        x_vals = np.arange(len(self.total_losses))
        plt.plot(x_vals, self.policy_losses, label="Policy Loss")
        plt.xlabel("Train Steps")
        plt.ylabel("Losses")
        plt.legend()
        plt.savefig(self.save_dir / "policy_loss.png")
        plt.show()


class TestResults(object):

    def __init__(self, save_dir=None):
        self.test_rewards = []
        self.save_dir = Path(save_dir)

    def add_reward(self, reward):
        self.test_rewards.append(reward)

    def last_n_average(self, n):
        last_n = self.test_rewards[-n:]
        l = len(last_n)
        s = sum(last_n)
        return s / l, l

    def plot_rewards(self):
        x_vals = np.arange(len(self.test_rewards))
        plt.plot(x_vals, self.test_rewards, label="Test Reward")
        plt.xlabel("Test Episodes")
        plt.ylabel("Reward")
        plt.savefig(self.save_dir / "test_rewards.png")
        plt.show()


class MinMaxStats(object):
    """
    A class that holds the min-max values of the tree.
    The MuZero MCTS has no knowledge of the game rewards and values, and in UCB
    selection, the Q value should be normalized to the appropriate scale.
    So, we keep a MinMaxStats for this purpose
    """

    def __init__(self, known_bounds: Optional[KnownBounds]):
        self.maximum = known_bounds.max if known_bounds else -MAX_FLOAT_VAL
        self.minimum = known_bounds.min if known_bounds else MAX_FLOAT_VAL

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MuZeroConfig(object):

    def __init__(self,
                 action_space_size: int,
                 max_moves: int,
                 discount: float,
                 dirichlet_alpha: float,
                 num_simulations: int,
                 batch_size: int,
                 buffer_size: int,
                 td_steps: int,
                 lr_init: float,
                 visit_softmax_temperature_fn,
                 num_epochs: int,
                 games_per_epoch: int,
                 train_per_epoch: int,
                 episodes_per_test: int,
                 known_bounds: Optional[KnownBounds] = None):

        # Self-Play
        self.action_space_size = action_space_size
        self.games_per_epoch = games_per_epoch
        self.num_epochs = num_epochs
        self.train_per_epoch = train_per_epoch
        self.episodes_per_test = episodes_per_test

        self.visit_softmax_temperature_fn = visit_softmax_temperature_fn
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.discount = discount

        # Root prior exploration noise.
        self.root_dirichlet_alpha = dirichlet_alpha
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # If we already have some information about which values occur in the
        # environment, we can use them to initialize the rescaling.
        # This is not strictly necessary, but establishes identical behaviour to
        # AlphaZero in board games.
        self.known_bounds = known_bounds

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_unroll_steps = 5
        self.td_steps = td_steps
        self.lr_init = lr_init


def get_cartpole_config(num_simulations=50):
    return MuZeroConfig(
        action_space_size=2,  # size of action space
        max_moves=200,  # number of moves in game
        discount=0.997,   # recommened 0.997
        dirichlet_alpha=0.1,  # exploration noise at root, very important
        num_simulations=num_simulations,  # number of MCTS rollouts
        batch_size=512,
        buffer_size=200,  # Small without reanalyze
        td_steps=10,  # TD rollout for value target
        lr_init=0.01,  # Initial Learning rate
        # known_bounds=KnownBounds(min=0, max=200), can be used
        # exploration in action selection
        visit_softmax_temperature_fn=visit_softmax_temperature,
        num_epochs=50,
        games_per_epoch=20,
        train_per_epoch=30,
        # games_per_epoch=2,
        # train_per_epoch=1,
        episodes_per_test=10
    )
