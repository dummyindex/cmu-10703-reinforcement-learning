# 403_muzero
# Homework created by Alex Singh and Robin Schmucker
MuZero HW


In this assignment you will implement a version of MuZero on a simple Cartpole environment.

The code is structured as follows:

main.py: This initializes the networks for training, the configuration for the environment, the replay buffer, and the environment itself.

self_play.py: In this file, the high level iterations of MuZero occur. In MuZero as presented in the paper, the algorithm is inherently distributed; multiple threads collect experience and add it to a replay buffer, while another thread continually trains the network with this experience. To avoid the difficulties with distributed training, for this assignment we only use a single thread, and alternate between collecting experience and updating the network.

Specifically, we iterate for num_epochs. For each epoch, we play games_per_epoch, followed by train_per_epoch. We then evaluate the network on episodes_per_test number of episodes.

config.py: In this file, we specify some util classes, as well as the general configuration for MuZero

game.py: In this file, we specify all helper functions for playing a game, and storing the final statistics after MCTS

mcts.py: In this file, we specify all functions needed for running MCTS from a root node

networks_base.py: In this file, we specify some abstract classes for holding all networks

networks.py: In this file, we specify the network architectures and training loop.

replay.py: In this file, we specify the replay buffer and associated sampling.

