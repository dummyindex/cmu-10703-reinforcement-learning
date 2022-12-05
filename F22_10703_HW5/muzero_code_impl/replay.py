import numpy as np
from collections import deque


class ReplayBuffer(object):

    def __init__(self, config):
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.td_steps = config.td_steps
        self.unroll_steps = config.num_unroll_steps

    def save_game(self, game):
        self.buffer.append(game)

    def sample_batch(self):
        """
        Sample a batch of experience.
        Sample batch_size games, along with an associated start position in each game
        Make the targets for the batch to be used in training
        """
        # Sample game according to max error
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [self.sample_position(g) for g in games]
        batch = []
        for (g, i) in zip(games, game_pos):
            targets, actions = g.make_target(
                i, self.unroll_steps, self.td_steps)
            batch.append(
                (g.state_history[i], actions, targets))
        state_batch, actions_batch, targets_batch = zip(*batch)
        actions_batch = list(zip(*actions_batch))
        targets_init_batch, *targets_recurrent_batch = zip(*targets_batch)
        batch = (state_batch, targets_init_batch, targets_recurrent_batch,
                 actions_batch)

        return batch

    def sample_game(self, p=None):
        """
        Picks a random game. This can be further adapted to prioritized experience
        replay
        """
        game = np.random.choice(self.buffer, p=p)
        return game

    def sample_position(self, game):
        """
        Sample a random position from the game to start unrolling
        """
        sampled_index = np.random.randint(
            len(game.reward_history)-self.unroll_steps)
        return sampled_index
