import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import util


@util.learner()
class BC(object):
    def __init__(
        self,
        state_dim : int,
        action_dim : int,
        lr : float = 1e-2,
        discount : float = 0.99
    ):

        self.actor = util.MLP(input_size=state_dim, 
                              output_size=action_dim, 
                              hidden_sizes=(256, 256), 
                              activation=nn.ReLU(),
                              final_activation=nn.Identity())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.discount = discount

    def select_action(self, state):
        return self.actor(state)

    def train(self, *data):
        state, action, next_state, reward, not_done = data
 
        # Compute actor loss
        loss = F.mse_loss(self.actor(state), action)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {'loss': loss}
