# 10-403 HW1 Problem 1
## Setup

You will need to install the openai gym and numpy in order to complete the
assignment. This assignment should work with Python 3.6 and above.

We have included a requirements.txt file to make the installation of
dependencies easy. We recommend (optional) making a virtualenv for this homework. If you
are not familiar with virtualenv you can read more about it here:
https://virtualenv.pypa.io/en/stable/

You can also install the [`virtualenvwrapper`](https://virtualenv.pypa.io/en/stable/) package if you want a more convenient command line interface.

To install the packages using pip and a virtualenv run the following commands:

```
virtualenv hw1q1_env
source hw1q1_env/bin/activate
pip install {package}
```

The following command should now work:

```
python -c 'import gym'
```

Note that you are free to work in a *conda env*, especially on AWS where the Deep Learning AMIs have most dependencies pre-installed in conda envs.


## OpenAI Gym Environments
### Creating the environments

To create the environment use the following code snippet:

```
import gym
import lake_envs

env = gym.make('Deterministic-4x4-FrozenLake-v0')
```

### Actions

There are four actions: LEFT, UP, DOWN, RIGHT represented as integers. The
`lake_envs` contains variables to reference
these. For example:

```
import lake_envs

print(lake_envs.LEFT)
```

will print out the number 0.

### Environment Attributes

This class contains the following important attributes:

- `nS` :: number of states
- `nA` :: number of actions
- `P` :: transitions, rewards, terminals

The `P` attribute will be the most important for your implementation of value
iteration and policy iteration. This attribute contains the model for the
particular map instance. It is a dictionary of dictionary of lists with the
following form:

```
P[s][a] = [(prob, nextstate, reward, is_terminal), ...]
```

For example, to get the probability of taking action LEFT in state 0 you would
use the following code:

```
import gym
import lake_envs

env = gym.make('Deterministic-4x4-FrozenLake-v0')
state = 0
action = lake_envs.LEFT
print(env.P[state][action])
```

This will print the list: `[(1.0, 0, 0.0, False)]` for the
`Deterministic-4x4-FrozenLake-v0` domain. There is one tuple in the list,
so there is only one possible next state.
- The next state will be state `0` (according to the second number in the
  tuple) with probability `1.0` (according to the first number in the tuple).
- The reward for this state-action pair is `0.0` according to the third number.
- The final tuple value `False` says that the next state is not terminal.

## Sample Code
### Running a random policy

See example.py for an example of how to run a random policy in the FrozenLake
environment:
```
python example.py
```
