# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gym
import lake_envs as lake_env

SYNC = "sync"
ASYNC_ORDERD = "async_ordered"
ASYNC_PERM = "async_perm"
ASYNC_MANHATTAN = "async_manhattan"


def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    print(str_policy)


def print_policy_grid(policy, action_names, ncols):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)

    for str_line in np.reshape(str_policy, [ncols, -1]):
        print(''.join(str_line))


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """
    # Hint: You might want to first calculate Q value,
    #       and then take the argmax.
    return policy


def evaluate_policy_sync(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """

    return evaluate_policy_general(env, value_func, gamma, policy, max_iterations, tol, is_sync=True)


def evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    return evaluate_policy_general(env, value_func, gamma, policy, max_iterations, tol, is_sync=True, heuristics="ordered")
    


def evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.

    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """

    return evaluate_policy_general(env, value_func, gamma, policy, max_iterations, tol, is_sync=False, heuristics="perm")


def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.

    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    policy_stable = True
    states = np.arange(len(value_func))
    for state in states:
        old_action = policy[state]
        action_vals = []
        for action in range(env.nA):
            new_val = 0
            # sample all possible next states and upate value function
            # though prob=1 only for one state (assume policy is deterministic)
            for transition_prob, nextstate, reward, is_terminal in env.P[state][action]:
                # # deterministic transition matrix, prob=1 for policy entries
                new_val += transition_prob * \
                    (reward + gamma * value_func[nextstate])
            action_vals.append(new_val)
        new_action = np.argmax(action_vals)
        if new_action != old_action:
            policy_stable = False
        policy[state] = new_action
    return policy_stable, policy


def evaluate_policy_general(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3, is_sync=True, heuristics="ordered"):
    """Performs policy evaluation.

    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    value_func: np.array
      The current value functione estimate
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """

    ordered_states = np.arange(len(value_func))

    num_steps = 0

    while True:
        num_steps += 1
        delta = 0
        if is_sync:
            new_val_func = np.array(value_func.copy())

        if heuristics == "ordered":
            heuristics_states = ordered_states
        elif heuristics == "perm":
            heuristics_states = np.random.permutation(ordered_states)
        else:
            raise NotImplemented

        for state in heuristics_states:
            val = value_func[state]
            action_vals = []

            new_val = 0
            # sample all possible next states and upate value function
            # though prob=1 only for one state (assume policy is deterministic)
            for action in range(env.nA):
                policy_prob = 1 if policy[state] == action else 0
                if policy_prob == 0:
                    continue
                for transition_prob, nextstate, reward, is_terminal in env.P[state][action]:
                    # # deterministic transition matrix, prob=1 for policy entries
                    new_val += policy_prob * transition_prob * \
                        (reward + gamma * value_func[nextstate])
                # print("<debug1> new_val: ", new_val)
            delta = max(delta, abs(new_val - value_func[state]))

            if is_sync:
                new_val_func[state] = new_val
            else:
                value_func[state] = new_val
        if is_sync:
            value_func = new_val_func
        if delta < tol:
            break
    return value_func, num_steps


def policy_iteration_general(env, gamma, max_iterations, tol, policy_type):
    """TODO """
    def init_value_func(env):
        # return np.random.sample(env.nS)
        return np.zeros(env.nS)

    def init_policy(env):
        # return np.random.randint(env.nA, size=env.nS)
        return np.zeros(env.nS, dtype=int)

    state_value_func = init_value_func(env)
    state_policy = init_policy(env)  # prob=1 for policy entries
    print("state_value_func", state_value_func)
    print("init policy: ", state_policy)
    assert state_value_func.shape == (env.nS,)

    total_policy_eval_steps = 0
    num_improvements = 0
    while True:
        # nextstate, reward, is_terminal, debug_info = env.step(
        #     env.action_space.sample())
        # env.render()
        if policy_type == SYNC:
            optimal_value_func, eval_iter_steps = evaluate_policy_sync(env, value_func=state_value_func, gamma=gamma,
                                                                       policy=state_policy, max_iterations=max_iterations, tol=tol)
        elif policy_type == ASYNC_ORDERD:
            optimal_value_func, eval_iter_steps = evaluate_policy_async_ordered(env, value_func=state_value_func, gamma=gamma,
                                                                                policy=state_policy, max_iterations=max_iterations, tol=tol)
        elif policy_type == ASYNC_PERM:
            optimal_value_func, eval_iter_steps = evaluate_policy_async_randperm(env, value_func=state_value_func, gamma=gamma,
                                                                                policy=state_policy, max_iterations=max_iterations, tol=tol)
        else:
            assert False
        total_policy_eval_steps += eval_iter_steps
        state_value_func = optimal_value_func

        policy_stable, new_policy = improve_policy(
            env, gamma, value_func=state_value_func, policy=state_policy)

        state_policy = new_policy
        num_improvements += 1
        if policy_stable:
            break
    return state_policy, state_value_func, num_improvements, total_policy_eval_steps


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    return policy_iteration_general(env, gamma, max_iterations, tol, policy_type=SYNC)


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    return policy_iteration_general(env, gamma, max_iterations, tol, policy_type=ASYNC_ORDERD)


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    return policy_iteration_general(env, gamma, max_iterations, tol, policy_type=ASYNC_PERM)



def get_goal_state(env):
    for state in range(env.nS):
        for action in range(env.nA):
            for trans in range(len(env.P[state][action])):
                prob_, ns_, r_, is_terminal = env.P[state][action][trans]
                if r_ >0.9 and is_terminal:
                    return ns_
    assert False, "goal state not found?!"


def manhatten_dist(s1, s2, nc):
    r1, c1 = s1 // nc, s1 % nc
    r2, c2 = s2 // nc, s2 % nc
    return abs(r1 - r2) + abs(c1 - c2)

def value_iteration_general(env, gamma, max_iterations=None, tol=None, is_sync=True, state_order="ordered"):
    ordered_states = np.arange(env.nS)
    actions = np.arange(env.nA)
    value_func = np.zeros(env.nS)  # initialize value function
    policy = np.zeros(env.nS, dtype=np.int)

    niters = 0
    goal_state = get_goal_state(env)

    while True:
      niters += 1
      if niters % 10000 == 0:
        print("niters:", niters)
      delta = 0
      if state_order == "ordered":
        states = ordered_states
      elif state_order == "perm":
        states = np.random.permutation(ordered_states)
      elif state_order == "manhattan":
        states = sorted(ordered_states, key=lambda s: manhatten_dist(s, goal_state, env.ncol))
      if is_sync:
        new_value_func = np.array(value_func.copy())
      for state in states:
        max_action_val = -np.inf
        max_action = None
        
        for action in actions:
          new_value = 0
          for p, nextstate, reward, _ in env.P[state][action]:
            new_value += p * (reward + gamma * value_func[nextstate])
            if max_action_val < new_value:
              max_action_val = new_value
              max_action = action

        delta = max(delta, abs(max_action_val - value_func[state]))
        if is_sync:
          new_value_func[state] = max_action_val
        else:
          value_func[state] = max_action_val

        # policy does not matter, always update regardless of sync or async
        policy[state] = max_action 
      if is_sync:
        value_func = new_value_func

      if delta < tol:
        break

    return value_func, niters, policy


def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    return value_iteration_general(env, gamma, max_iterations=max_iterations, tol=tol, is_sync=True, state_order="ordered")


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    return value_iteration_general(env, gamma, max_iterations=max_iterations, tol=tol, is_sync=False, state_order="ordered")


def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    return value_iteration_general(env, gamma, max_iterations=max_iterations, tol=tol, is_sync=False, state_order="perm")


def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    return value_iteration_general(env, gamma, max_iterations=max_iterations, tol=tol, is_sync=False, state_order="manhattan")


######################
#  Optional Helpers  #
######################

# Here we provide some helper functions simply for your convinience.
# You DON'T necessarily need them, especially "env_wrapper" if
# you want to deal with it in your different ways.

# Feel FREE to change/delete these helper functions.

def display_policy_letters(env, policy):
    """Displays a policy as letters, as required by problem 1.2 & 1.3

    Parameters
    ----------
    env: gym.core.Environment
    policy: np.ndarray, with shape (env.nS)
    """
    policy_letters = []
    for l in policy:
        policy_letters.append(lake_env.action_names[l][0])

    policy_letters = np.array(policy_letters).reshape(env.nrow, env.ncol)

    for row in range(env.nrow):
        print(''.join(policy_letters[row, :]))


def env_wrapper(env_name):
    """Create a convinent wrapper for the loaded environment

    Parameters
    ----------
    env: gym.core.Environment

    Usage e.g.:
    ----------
        envd4 = env_load('Deterministic-4x4-FrozenLake-v0')
        envd8 = env_load('Deterministic-8x8-FrozenLake-v0')
    """
    env = gym.make(env_name)

    # T : the transition probability from s to s’ via action a
    # R : the reward you get when moving from s to s' via action a
    env.T = np.zeros((env.nS, env.nA, env.nS))
    env.R = np.zeros((env.nS, env.nA, env.nS))

    for state in range(env.nS):
        for action in range(env.nA):
            for prob, nextstate, reward, is_terminal in env.P[state][action]:
                env.T[state, action, nextstate] = prob
                env.R[state, action, nextstate] = reward
    return env


def value_func_heatmap(env, value_func):
    """Visualize a policy as a heatmap, as required by problem 1.2 & 1.3

    Note that you might need:
        import matplotlib.pyplot as plt
        import seaborn as sns

    Parameters
    ----------
    env: gym.core.Environment
    value_func: np.ndarray, with shape (env.nS)
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(np.reshape(value_func, [env.nrow, env.ncol]),
                annot=False, linewidths=.5, cmap="GnBu_r", ax=ax,
                yticklabels=np.arange(1, env.nrow+1)[::-1],
                xticklabels=np.arange(1, env.nrow+1))
    # Other choices of cmap: YlGnBu
    # More: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    return None


if __name__ == "__main__":
    envs = ['Deterministic-4x4-FrozenLake-v0',
            'Deterministic-8x8-FrozenLake-v0']
    # Define num_trials, gamma and whatever variables you need below.
