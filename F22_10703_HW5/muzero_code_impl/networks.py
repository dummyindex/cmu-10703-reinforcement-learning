import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MSE
from tensorflow.keras.regularizers import L2
import numpy as np
import math
from typing import Callable
from networks_base import BaseNetwork
from config import MuZeroConfig


def action_to_one_hot(action, action_space_size):
    """
    Compute one hot of action to be combined with state representation
    """
    return np.array([1 if i == action else 0 for i in range(action_space_size)]).reshape(1, -1)


class CartPoleNetwork(BaseNetwork):

    def __init__(self, action_size, state_shape, embedding_size, max_value):
        """
        Defines the CartPoleNetwork
        action_size: the number of actions
        state_shape: the shape of the input state
        embedding_size: the size of the embedding layer for representation
        max_value: denotes the max reward of a game for value transform
        """
        self.action_size = action_size
        self.state_shape = state_shape
        self.embedding_size = embedding_size
        self.train_steps = 0
        self.hidden_neurons = 48
        # value support size should be <= math.ceil(math.sqrt(max_value)) + 1
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1
        # Only in negative values are allowed
        self.full_support_size = 2 * self.value_support_size + 1
        regularizer = L2(1e-4)  # Weight Decay Strength

        # Create Networks, we use tanh activation as a normalizer on the state
        # Standard muzero performs max-min normalization
        representation_network = Sequential([Dense(self.hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                             Dense(self.embedding_size, activation='tanh',
                                                   kernel_regularizer=regularizer)])
        # Change to full support if negative values allowed
        value_network = Sequential([Dense(self.hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                    Dense(self.value_support_size, kernel_regularizer=regularizer)])

        policy_network = Sequential([Dense(self.hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                     Dense(action_size, kernel_regularizer=regularizer)])
        dynamic_network = Sequential([Dense(self.hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                      Dense(self.embedding_size, activation='tanh',
                                            kernel_regularizer=regularizer)])
        reward_network = Sequential([Dense(16, activation='relu', kernel_regularizer=regularizer),
                                     Dense(1, kernel_regularizer=regularizer)])

        super().__init__(representation_network, value_network,
                         policy_network, dynamic_network, reward_network)

    def training_steps(self):
        return self.train_steps

    def _value_transform(self, value_support) -> float:
        value = self._softmax(value_support)
        # Change to -value_support_size -> value_support_size for full support
        value = np.dot(value, range(self.value_support_size))
        value = tf.math.sign(value) * (
                ((tf.math.sqrt(1 + 4 * 0.001
                 * (tf.math.abs(value) + 1 + 0.001)) - 1) / (2 * 0.001)) ** 2 - 1
        )
        return value.numpy()[0]

    def _scalar_to_support(self, target_value):
        batch = len(target_value)
        targets = np.zeros((batch, self.value_support_size))
        target_value = tf.math.sign(target_value) * \
            (tf.math.sqrt(tf.math.abs(target_value) + 1)
             - 1 + 0.001*target_value)
        target_value = tf.clip_by_value(
            target_value, 0, self.value_support_size)
        floor = tf.math.floor(target_value)
        rest = target_value - floor
        targets[range(batch), tf.cast(floor, tf.int32)] = 1 - rest
        indexes = tf.cast(floor, tf.int32) + 1
        mask = indexes < self.value_support_size
        batch_mask = tf.boolean_mask(range(batch), mask)
        rest_mask = tf.boolean_mask(rest, mask)
        index_mask = tf.boolean_mask(indexes, mask)
        targets[batch_mask, index_mask] = rest_mask
        return targets

    def _reward_transform(self, reward) -> float:
        """
        No reward transform for cartpole
        """
        # return np.asscalar(reward.numpy()[0])
        # TODO: discuss asscalar equivalence
        return reward.numpy()[0].item()

    def _conditioned_hidden_state(self, hidden_state: np.array, action: int) -> np.array:
        """
        concatenate the hidden state and action for input to recurrent model
        """
        conditioned_hidden = tf.concat(
            (hidden_state, action_to_one_hot(action, self.action_size)), axis=1)
        return conditioned_hidden

    def _softmax(self, values):
        """
        Compute softmax
        """
        return tf.nn.softmax(values)

    def get_value_target(self, state):
        return self._value_transform(self.target_network.__call__(state))

    def cb_get_variables(self) -> Callable:
        """
        Return a callback that return the trainable variables of the network.
        """

        def get_variables():
            networks = (self.representation_network, self.value_network, self.policy_network,
                        self.dynamic_network, self.reward_network)
            return [variables
                    for variables_list in map(lambda n: n.trainable_weights, networks)
                    for variables in variables_list]

        return get_variables

    def save(self, path):
        """Save the networks."""
        if not os.path.isdir(path):
            os.mkdir(path)

        self.representation_network.save_weights(
            path + "/representation_net")  # , save_format='h5py')
        self.value_network.save_weights(
            path + "/value_net")  # , save_format='h5py')
        self.policy_network.save_weights(
            path + "/policy_net")  # , save_format='h5py')
        self.dynamic_network.save_weights(
            path + "/dynamic_net")  # , save_format='h5py')
        self.reward_network.save_weights(
            path + "/reward_net")  # , save_format='h5py')
        print("saved network at path:", path)

    def load(self, path):
        """Load previously stored network parameters."""
        self.built = True
        self.representation_network.built = True
        self.representation_network.load_weights(path + "/representation_net")
        self.value_network.load_weights(path + "/value_net")
        self.policy_network.load_weights(path + "/policy_net")
        self.dynamic_network.load_weights(path + "/dynamic_net")
        self.reward_network.load_weights(path + "/reward_net")
        print("loaded pre-trained weights at path:", path)


def scale_gradient(tensor, scale):
    """
    Function to scale gradient as described in MuZero Appendix
    """
    return tensor*scale + tf.stop_gradient(tensor) * (1. - scale)


def train_network(config, network, replay_buffer, optimizer, train_results):
    """
    Train Network for N steps
    """
    for _ in range(config.train_per_epoch):
        batch = replay_buffer.sample_batch()
        update_weights(config, network, optimizer, batch, train_results)


def update_weights(config: MuZeroConfig, network: CartPoleNetwork, optimizer, batch, train_results):
    """
    TODO: Implement this function
    Train the network_model by sampling games from the replay_buffer.
    config: A dictionary specifying parameter configurations
    network: The network class to train
    optimizer: The optimizer used to update the network_model weights
    batch: The batch of experience
    train_results: The class to store the train results

    Hints:
    The network initial_model should be used to create the hidden state
    The recurrent_model should be used as the dynamics, which unroll in the latent space.

    You should accumulate loss in the value, the policy, and the reward (after the first state)
    Loss Note: The policy outputs are the logits, same with the value categorical representation
    You should use tf.nn.softmax_cross_entropy_with_logits to compute the loss in these cases
    """
    # for every game in sample batch, unroll and update network_model weights
    def loss():
        loss = 0
        total_value_loss = 0
        total_reward_loss = 0
        total_policy_loss = 0
        (state_batch, targets_init_batch, targets_recurrent_batch,
         actions_batch) = batch

        # YOUR CODE HERE: Perform initial embedding of state batch

        # pred_values, transformed_rewards, policy_logits, hidden_representation = network.initial_inference(np.array(state_batch))
        init_latent_state, pred_values, policy_logits = network.initial_model.call(np.array(state_batch))
        
        target_value_batch, _, target_policy_batch = zip(
            *targets_init_batch)
        # Use this to convert scalar value targets to categorical representation
        # This now matches output of initial_model, and can be used with cross entropy loss
        target_value_batch = network._scalar_to_support(
            tf.convert_to_tensor(target_value_batch))
        # YOUR CODE HERE: Compute the loss of the first pass (no reward loss)
        # Remember to scale value loss!
        l_v_init = tf.nn.softmax_cross_entropy_with_logits(labels=target_value_batch, logits=pred_values)
        l_p_init = tf.nn.softmax_cross_entropy_with_logits(labels=target_policy_batch, logits=policy_logits)
        l_init = tf.reduce_mean(l_v_init) * 0.25 + tf.reduce_mean(l_p_init)

        loss = l_init
        mse_loss = tf.keras.losses.MeanSquaredError()
        # TODO: discuss: according to https://piazza.com/class/l6ux8qcfetf38o/post/467, use config val
        # num_unroll_step = len(actions_batch)
        cur_latent_state = init_latent_state
        for actions_batch, targets_batch in zip(actions_batch, targets_recurrent_batch):
            target_value_batch, target_reward_batch, target_policy_batch = zip(
                *targets_batch)
            # YOUR CODE HERE:
            # Create conditioned_representation: concatenate representations with actions batch
            # Recurrent step from conditioned representation: recurrent + prediction networks

            # TODO: double check this
            # pred_values, transformed_rewards, policy_logits, hidden_representation = network.recurrent_inference(np.array(state_batch), np.array(actions_batch))
            actions_batch_onehot = tf.one_hot(actions_batch, config.action_space_size)
            recurrent_input = tf.concat((cur_latent_state, actions_batch_onehot), axis=1)
            new_latent_state,  transformed_rewards, pred_values, policy_logits = network.recurrent_model(recurrent_input)
            # Same as above, convert scalar targets to categorical
            target_value_batch = tf.convert_to_tensor(target_value_batch)
            target_value_batch = network._scalar_to_support(target_value_batch)

            target_policy_batch = tf.convert_to_tensor(target_policy_batch)
            target_reward_batch = tf.convert_to_tensor(target_reward_batch)

            # YOUR CODE HERE: Compute value loss, reward loss, policy loss
            # Remember to scale value loss!
            # Add to total losses
            l_v = tf.nn.softmax_cross_entropy_with_logits(target_value_batch, pred_values)
            l_r = mse_loss(target_reward_batch, transformed_rewards)
            l_p = tf.nn.softmax_cross_entropy_with_logits(target_policy_batch, policy_logits)
            # reduce with mean
            l_v, l_r, l_p = tf.reduce_mean(l_v), tf.reduce_mean(l_r), tf.reduce_mean(l_p)
            l_step = 0.25 * l_v + l_r + l_p
            total_policy_loss += l_p
            total_value_loss += l_v
            total_reward_loss += l_r
            # YOUR CODE HERE: Half the gradient of the representation
            # TODO: discussed during recitation
            scale_gradient(cur_latent_state, 0.5)
            # YOUR CODE HERE: Sum the losses, scale gradient of the loss, add to overall loss
            loss += l_step / config.num_unroll_steps
            cur_latent_state = new_latent_state
        
        train_results.total_losses.append(loss)
        train_results.value_losses.append(total_value_loss)
        train_results.policy_losses.append(total_policy_loss)
        train_results.reward_losses.append(total_reward_loss)
        return loss
    
    optimizer.minimize(loss=loss, var_list=network.cb_get_variables())
    network.train_steps += 1
    # raise NotImplementedError()
