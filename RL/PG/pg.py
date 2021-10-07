import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_discounted_cumsum import discounted_cumsum_right
from torch.distributions.categorical import Categorical
from util import MLP, module
    

@module
class PG(object):
    def __init__(
        self,
        state_dim : int,
        action_dim : int,
        discount : float = 0.97,
        tau : float = 0.05,
        eps : float = 1e-8
    ):
        self.actor = MLP(input_size=state_dim, 
                         output_size=action_dim, 
                         hidden_sizes=(256, 256), 
                         activation=nn.ReLU(),
                         final_activation=nn.LogSoftmax(dim=-1))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.eps = eps
        self.action_dim = action_dim

        self.total_it = 0

    def select_action(self, state, deterministic=True):
        if deterministic:
            return torch.argmax(self.actor(state), -1).numpy()
        else:
            return Categorical(logits=self.actor(state)).sample().numpy()

    def train(self, *data):
        self.total_it += 1
        state, action, next_state, reward, not_done = data
        
        # Estimate Q-Values with MC samples
        qval = discounted_cumsum_right(torch.unsqueeze(reward * not_done, axis=1), self.discount)

        # Get current policy
        policy = self.actor(state)
 
        # Compute surrogate objective
        loss = -torch.sum(qval * policy)
        losses = {'loss': loss}

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        losses


    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.critic)