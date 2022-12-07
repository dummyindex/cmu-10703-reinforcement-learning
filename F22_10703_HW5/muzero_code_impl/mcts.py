from typing import List
import numpy as np
from networks_base import BaseNetwork
import tensorflow as tf


class Node(object):

    def __init__(self, prior):
        """
        Node in MCTS
        prior: The prior on the node, computed from policy network
        """
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_representation = None
        self.reward = 0
        self.expanded = False

    def value(self):
        """
        Compute value of a node
        """
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count


def run_mcts(config, root, network, min_max_stats):
    """
    Main loop for MCTS for config.num_simulations simulations

    root: the root node
    network: the network
    min_max_stats: the min max stats object for the simulation

    Hint:
    The MCTS should capture selection, expansion and backpropagation
    """
    for i in range(config.num_simulations):
        history = []
        node = root
        search_path = [node]

        while node.expanded:
            action, node = select_child(config, node, min_max_stats)
            history.append(action)
            search_path.append(node)
        parent = search_path[-2]
        action = history[-1]
        value = expand_node(node, list(
            range(config.action_space_size)), network, parent.hidden_representation, action)
        backpropagate(search_path, value,
                      config.discount, min_max_stats)


def select_action(config, num_moves, node: Node, network, test=False):
    """
    Select an action to take

    If in train mode: action selection should be performed stochastically
    with temperature t
    If in test mode: action selection should be performed with argmax
    """
    visit_counts = [
        (child.visit_count, action) for action, child in node.children.items()
    ]
    if not test:
        t = config.visit_softmax_temperature_fn(num_moves=num_moves)
        action = softmax_sample(visit_counts, t)
    else:
        action = softmax_sample(visit_counts, 0)
    return action


def select_child(config, node: Node, min_max_stats):
    """
    TODO: Implement this function
    Select a child in the MCTS
    This should be done using the UCB score, which uses the
    normalized Q values from the min max stats
    """

    ucb_scores = []
    children = []
    actions = []
    for action, child in node.children.items():
        _ucb_score = ucb_score(config, node, child, min_max_stats)
        ucb_scores.append(_ucb_score)
        children.append(child)
        actions.append(action)
    max_idx = np.argmax(ucb_scores)

    # TODO: check with teammates: cartpole is deterministic (aka, f: (a, s) -> s' is deterministic)
    action = actions[max_idx]
    child = node.children[action]
    return action, child


def ucb_score(config, parent:Node, child: Node, min_max_stats):
    """
    Compute UCB Score of a child given the parent statistics
    """
    pb_c = np.log((parent.visit_count + config.pb_c_base + 1)
                  / config.pb_c_base) + config.pb_c_init
    pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c*child.prior
    if child.visit_count > 0:
        value_score = min_max_stats.normalize(
            child.reward + config.discount*child.value())
    else:
        value_score = 0
    return prior_score + value_score


def expand_root(node: Node, actions: list, network: BaseNetwork, current_state):
    """
    TODO: Implement this function
    Expand the root node given the current state

    This should perform initial inference, and calculate a softmax policy over children
    You should set the attributes hidden representation, the reward, the policy and children of the node
    Also, set node.expanded to be true
    For setting the nodes children, you should use node.children and  instantiate
    with the prior from the policy
    Args:
        action: list(range(config.action_space_size))

    Return: the value of the root
    """
    # get hidden state representation
    transformed_value, reward, policy_logits, hidden_representation = network.initial_inference(current_state.reshape(1, -1))
    node.reward = reward
    node.hidden_representation = hidden_representation
    # Extract softmax policy and set node.policy
    policy = tf.nn.softmax(policy_logits)
    # TODO: not used anywhere, see https://piazza.com/class/l6ux8qcfetf38o/post/473, just stop saving node.policy
    # node.policy = policy

    # instantiate node's children with prior values, obtained from the predicted policy
    policy_arr = policy.numpy()[0]
    for action in actions: # action is int
        node.children[action] = Node(policy_arr[action])

    # set node as expanded
    node.expanded = True
    return transformed_value


def expand_node(node: Node, actions, network: BaseNetwork, parent_state, parent_action):
    """
    TODO: Implement this function
    Expand a node given the parent state and action
    This should perform recurrent_inference, and store the appropriate values
    The function should look almost identical to expand_root

    Return: value
    """
    # get hidden state representation
    transformed_value, transformed_rewards, policy_logits, hidden_representation = network.recurrent_inference(tf.reshape(parent_state, (1, -1)), tf.reshape(parent_action, (1, -1)))
    node.reward = transformed_rewards
    node.hidden_representation = hidden_representation

    policy_logits = policy_logits.numpy()[0]
    for action in actions:
        node.children[action] = Node(policy_logits[action])
    
    node.expanded = True
    return transformed_value


def backpropagate(path: List[Node], value, discount, min_max_stats):
    """
    Backpropagate the value up the path

    This should update a nodes value_sum, and its visit count

    Update the value with discount and reward of node
    """
    cur_sum = value
    for node in reversed(path):
        # YOUR CODE HERE
        node.visit_count += 1

        # TODO: discuss correctness and meaning of value_sum; below maybe incorrect
        node.value_sum += cur_sum
        cur_sum = (discount * cur_sum + node.reward) # * node.prior
        min_max_stats.update(node.value())


def add_exploration_noise(config, node):
    """
    Add exploration noise by adding dirichlet noise to the prior over children
    This is governed by root_dirichlet_alpha and root_exploration_fraction
    """
    actions = list(node.children.keys())
    noise = np.random.dirichlet([config.root_dirichlet_alpha]*len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1-frac) + n*frac


def visit_softmax_temperature(num_moves):
    """
    This function regulates exploration vs exploitation when selecting actions
    during self-play.
    Given the current number of games played by the learning algorithm, return the
    temperature value to be used by MCTS.

    You are welcome to devise a more complicated temperature scheme
    """
    return 1


def softmax_sample(visit_counts_action, temperature):
    """
    Sample an action

    Input: visit_counts as list of [(visit_count, action)] for each child
    If temperature == 0, choose argmax
    Else: Compute distribution over visit_counts and sample action as in writeup
    """

    # YOUR CODE HERE
    if temperature == 0:
        idx = np.argmax([count for count, _ in visit_counts_action])
        return visit_counts_action[idx][1]
    else:
        config_z = np.array([count for count, _ in visit_counts_action])** (1/temperature)
        probs = config_z / np.sum(config_z)
        action_idx = np.random.choice(len(probs), p=probs)
        return visit_counts_action[action_idx][1]
