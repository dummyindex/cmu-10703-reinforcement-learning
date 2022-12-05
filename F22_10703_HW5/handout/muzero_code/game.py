import itertools
from mcts import Node
import numpy as np


# discretized action space for walker
DISC = [-1, 0.0, 1]
CONVERTER = {}
for i, v in enumerate(itertools.product(DISC, DISC, DISC, DISC)):
    CONVERTER[i] = np.array(v)


class Game:

    def __init__(self, action_space_size, discount, curr_state):
        """
        Game class
        action_space_size: number of actions
        discount: discount factor
        curr_state: the start state of the game
        """

        self.action_space_size = action_space_size
        self.curr_state = curr_state
        self.done = False
        self.discount = discount
        self.priorities = None

        self.state_history = [self.curr_state]
        self.action_history = []
        self.reward_history = []

        self.root_values = []
        self.child_visits = []

    def store_search_statistics(self, root: Node):
        """
        Stores the search statistics for the root node

        1. Stores the root node value, computed from the MCTS
        2. Stores the normalized root node child visits, this is the POLICY target
        """
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append(np.array([
            root.children[a].visit_count
            / sum_visits if a in root.children else 0
            for a in range(self.action_space_size)
        ]))
        self.root_values.append(root.value())

    def action(self, action, env):
        obs, reward, done, _ = env.step(action)
        # Only for walker environment
        # obs, reward, done, _ = env.step(CONVERTER[action])
        self.curr_state = obs
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.done = done
        if not done:
            self.state_history.append(self.curr_state)

    def compute_bootstrap(self, index, td_steps):
        value = self.root_values[index] * (self.discount**td_steps)
        return value

    def make_target(self, state_index, num_unroll_steps, td_steps):
        """
        Makes the targets for training

        state_index: the start state
        num_unroll_steps: how many times to unroll from the current state
                          each unroll forms a new target
        td_steps: the number of td steps used in bootstrapping the value function

        Hint: if the number of td_steps goes beyond the game length, the bootstrapped value is 0
        Hint: States past the end of the game should be treated as absorbing states
        Hint: The reward target should be the reward from the last step
        """
        targets = []
        actions = []

        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps
            if bootstrap_index < len(self.root_values):
                value = self.compute_bootstrap(bootstrap_index, td_steps)
            else:
                value = 0

            for i, reward in enumerate(self.reward_history[current_index:bootstrap_index]):
                value += reward * (self.discount**i)

            if current_index > 0 and current_index <= len(self.reward_history):
                last_reward = self.reward_history[current_index-1]
            else:
                last_reward = 0

            if current_index < len(self.root_values):
                targets.append((value, last_reward,
                                self.child_visits[current_index]))
                actions.append(self.action_history[current_index])
            else:
                assert 0 == 1
                # States past the end of games are treated as absorbing states.
                num_actions = self.action_space_size
                targets.append(
                    (0, last_reward, np.array([1.0 / num_actions for _ in range(num_actions)])))
                actions.append(np.random.choice(num_actions))
        return targets, actions
