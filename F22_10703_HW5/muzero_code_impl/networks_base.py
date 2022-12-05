from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from tensorflow.keras.models import Model


class AbstractNetwork(ABC):

    def __init__(self):
        self.train_steps = 0

    @abstractmethod
    def initial_inference(self, image):
        pass

    @abstractmethod
    def recurrent_inference(self, hidden_state, action):
        pass


class InitialModel(Model):
    """
    Model that combine the representation and prediction (value+policy) network.
    You should use this in training loop
    """

    def __init__(self, representation_network: Model, value_network: Model, policy_network: Model):
        super(InitialModel, self).__init__()
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network

    def call(self, state):
        hidden_representation = self.representation_network(state)
        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)
        return hidden_representation, value, policy_logits


class RecurrentModel(Model):
    """
    Model that combine the dynamic, reward and prediction (value+policy) network.
    You should use this in training loop
    """

    def __init__(self, dynamic_network: Model, reward_network: Model, value_network: Model, policy_network: Model):
        super(RecurrentModel, self).__init__()
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network
        self.value_network = value_network
        self.policy_network = policy_network

    def call(self, conditioned_hidden):
        hidden_representation = self.dynamic_network(conditioned_hidden)
        reward = self.reward_network(conditioned_hidden)
        value = self.value_network(hidden_representation)
        policy_logits = self.policy_network(hidden_representation)
        return hidden_representation, reward, value, policy_logits


class BaseNetwork(AbstractNetwork):
    """
    Base class that contains all the networks and models of MuZero.
    """

    def __init__(self, representation_network: Model, value_network: Model, policy_network: Model,
                 dynamic_network: Model, reward_network: Model):
        super().__init__()
        # Networks blocks
        self.representation_network = representation_network
        self.value_network = value_network
        self.policy_network = policy_network
        self.dynamic_network = dynamic_network
        self.reward_network = reward_network

        # Models for inference and training
        self.initial_model = InitialModel(
            self.representation_network, self.value_network, self.policy_network)
        self.recurrent_model = RecurrentModel(self.dynamic_network, self.reward_network, self.value_network,
                                              self.policy_network)

    def initial_inference(self, state: np.array):
        """
        representation + prediction function
        Initial Inference produces 0 reward
        """

        hidden_representation, value, policy_logits = self.initial_model.__call__(
            state)
        return self._value_transform(value), 0, policy_logits, hidden_representation

    def recurrent_inference(self, hidden_state: np.array, action: int):
        """
        dynamics + prediction function
        """

        conditioned_hidden = self._conditioned_hidden_state(
            hidden_state, action)
        hidden_representation, reward, value, policy_logits = self.recurrent_model.__call__(
            conditioned_hidden)

        return self._value_transform(value), self._reward_transform(reward), policy_logits, hidden_representation

    @abstractmethod
    def _value_transform(self, value) -> float:
        pass

    @abstractmethod
    def _reward_transform(self, reward) -> float:
        pass

    @abstractmethod
    def _conditioned_hidden_state(self, hidden_state: np.array, action: int) -> np.array:
        pass

    def cb_get_variables(self) -> Callable:
        """Return a callback that return the trainable variables of the network."""

        def get_variables():
            networks = (self.representation_network, self.value_network, self.policy_network,
                        self.dynamic_network, self.reward_network)
            return [variables
                    for variables_list in map(lambda n: n.trainable_weights, networks)
                    for variables in variables_list]

        return get_variables
