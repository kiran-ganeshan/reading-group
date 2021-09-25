import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils import MLP
    

class PG(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        discount=0.99,
        tau=0.005,
        eps=1e-9
    ):

        self.actor = MLP(input_size=state_dim, 
                         output_size=action_dim, 
                         hidden_sizes=(256, 256), 
                         activation=nn.ReLU(),
                         final_activation=nn.LogSoftmax())
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.eps = eps
        self.action_dim = action_dim

        self.total_it = 0

    def select_action(self, state, deterministic=True):
        if deterministic:
            return torch.argmax(self.actor(state), -1)
        else:
            return Categorical(logits=self.actor(state)).sample()

    def train(self, *data):
        self.total_it += 1
        state, action, next_state, reward, q_val, not_done = data

        # Get current policy
        policy = self.actor(state)
 
        # Compute surrogate objective
        loss = -q_val * policy
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