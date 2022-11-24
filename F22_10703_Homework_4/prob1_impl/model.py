import os
import numpy as np
import torch
import torch.nn as nn
import operator
from functools import reduce
from utils.util import ZFilter
import torch.nn.functional as F

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

import logging
log = logging.getLogger('root')


class PENN(nn.Module):
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate, device=None):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        super().__init__()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Log variance bounds
        self.max_logvar = torch.tensor(-3 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device)
        self.min_logvar = torch.tensor(-7 * np.ones([1, self.state_dim]), dtype=torch.float, device=self.device)

        # Create or load networks
        self.networks = nn.ModuleList([self.create_network(n) for n in range(self.num_nets)]).to(device=self.device)
        self.opt = torch.optim.Adam(self.networks.parameters(), lr=learning_rate)

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, device=self.device, dtype=torch.float)
        return [self.get_output(self.networks[i](inputs)) for i in range(self.num_nets)]

    def get_output(self, output):
        """
        Argument:
          output: the raw output of a single ensemble member
        Return:
          mean and log variance
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - nn.functional.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + nn.functional.softplus(logvar - self.min_logvar)
        return mean, logvar

    def get_loss(self, target, mean, logvar):
        """nll loss for a single network"""
        # TODO: write your code here
        # total_avg_loss = 0
        # for net_idx in range(self.num_nets):
        #     # TODO: constant term: + 1/2 * log(2*pi)? necessary?
        #     # scale by two (note we have logvar here)
        #     # loss = torch.sum(((target - mean[net_idx]) ** 2) / torch.exp(logvar[net_idx])\
        #     #     + logvar[net_idx], dim=1)
        #     # loss = torch.mean(loss)

        #     loss = F.gaussian_nll_loss
        #     total_avg_loss += loss
        # return total_avg_loss / self.num_nets
        loss = F.gaussian_nll_loss(mean, target, torch.exp(logvar))
        return loss

    def create_network(self, n):
        layer_sizes = [self.state_dim + self.action_dim, HIDDEN1_UNITS, HIDDEN2_UNITS, HIDDEN3_UNITS]
        layers = reduce(operator.add,
                        [[nn.Linear(a, b), nn.ReLU()]
                         for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], 2 * self.state_dim)]
        return nn.Sequential(*layers)

    def train_model(self, inputs, targets, batch_size=128, num_train_itrs=5):
        """
        Training the Probabilistic Ensemble (Algorithm 2)
        Argument:
          inputs: state and action inputs. Assumes that inputs are standardized.
          targets: delta states
        Return:
            List containing the average loss of all the networks at each train iteration

        """
        # TODO: write your code here
        self.train()
        step_avg_loss = []
        for iter in range(num_train_itrs):
            for i in range(0, inputs.shape[0], batch_size):
                batch_inputs = torch.Tensor(inputs[i:i + batch_size]).float()
                batch_targets = torch.Tensor(targets[i:i + batch_size]).float()
                # means, vars = [], []
                step_loss_temp = 0
                for net in self.networks:
                    net.zero_grad()
                    raw_output = net(batch_inputs)
                    mean, logvar = self.get_output(raw_output)
                    loss = self.get_loss(batch_targets, mean, logvar)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    step_loss_temp += loss.item()
                step_avg_loss.append(step_loss_temp / self.num_nets)
        return step_avg_loss

        

        