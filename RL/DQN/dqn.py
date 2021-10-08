import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import ActionType, learner, MLP, one_hot


@learner("critic")
class DQN(object):
    def __init__(
        self,
        state_dim : int,
        action_dim : int,
        lr : float = 3e-6,
        discount : float = 0.99,
        tau : float = 0.005,
        eps : float = 1e-2
    ):

        self.critic = MLP(input_size=state_dim, 
                          output_size=action_dim, 
                          hidden_sizes=(30, 30), 
                          activation=nn.ReLU(),
                          final_activation=nn.Identity())
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.discount = discount
        self.tau = tau
        self.eps = eps
        self.action_dim = action_dim

        self.total_it = 0

    def select_action(self, state):
        if np.random.uniform(0, 1) < self.eps:
            return np.random.randint(0, self.action_dim)
        else:
            return torch.argmax(self.critic(state), -1)

    def train(self, *data):
        self.total_it += 1
        state, action, next_state, reward, not_done = data
        
        # Compute the target Q value
        with torch.no_grad(): 
            target_Q = self.critic_target(next_state)
            target_Q = torch.max(target_Q, -1)[0]
            data = {'next_q': target_Q}
            target_Q = reward + self.discount * not_done * target_Q
            data = {'target_q': target_Q, **data}
            
        # Get current Q estimates
        q_mask = one_hot(action, num_classes=self.action_dim)
        Q = torch.sum(self.critic(state) * q_mask, dim=-1)
 
        # Compute critic loss
        critic_loss = F.mse_loss(Q, target_Q)
        losses = {'loss': critic_loss}

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return losses


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
