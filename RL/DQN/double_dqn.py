import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import util


@util.learner("critic1", "critic2")
class DoubleDQN(object):
    def __init__(
        self,
        state_dim : int,
        action_dim : int,
        beta1 : float,
        beta2 : float,
        weight_decay : float,
        lr : float = 1e-2,
        discount : float = 0.99,
        tau : float = 0.005,
        eps : float = 1e-3
    ):

        self.critic1 = util.MLP(input_size=state_dim, 
                                output_size=action_dim, 
                                hidden_sizes=(256, 256), 
                                activation=nn.ReLU(),
                                final_activation=nn.Identity())
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), 
                                                  lr=lr, 
                                                  betas=(beta1, beta2), 
                                                  weight_decay=weight_decay)
        
        self.critic2 = util.MLP(input_size=state_dim, 
                                output_size=action_dim, 
                                hidden_sizes=(256, 256), 
                                activation=nn.ReLU(),
                                final_activation=nn.Identity())
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), 
                                                  lr=lr,
                                                  betas=(beta1, beta2), 
                                                  weight_decay=weight_decay)

        self.discount = discount
        self.tau = tau
        self.eps = eps
        self.action_dim = action_dim

        self.total_it = 0

    def select_action(self, state):
        if np.random.uniform(0, 1) < self.eps:
            return np.random.randint(0, self.action_dim)
        else:
            return torch.argmax(self.critic1(state), -1)

    def train(self, *data):
        self.total_it += 1
        state, action, next_state, reward, not_done = data
        
        # Compute the target Q value
        with torch.no_grad(): 
            target_Q1 = self.critic1_target(next_state)
            target_Q2 = self.critic2_target(next_state)
            argmax_mask = util.one_hot(torch.argmax(target_Q2, -1), 
                                       num_classes=target_Q2.shape[-1])
            target_Q = torch.sum(target_Q1 * argmax_mask, -1)[0]
            data = {'next_q': target_Q}
            target_Q = reward + self.discount * not_done * target_Q
            data = {'target_q': target_Q, **data}
            
        # Get current Q estimates
        q_mask = util.one_hot(action, num_classes=self.action_dim)
        Q1 = torch.sum(self.critic1(state) * q_mask, dim=-1)
        Q2 = torch.sum(self.critic2(state) * q_mask, dim=-1)
 
        # Compute critic loss
        critic1_loss = F.mse_loss(Q1, target_Q)
        critic2_loss = F.mse_loss(Q2, target_Q)
        loss = critic1_loss + critic2_loss
        losses = {'critic1_loss': critic1_loss, 'critic2_loss': critic2_loss, 'loss': loss}

        # Optimize the critic
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()

        # Update the frozen target models
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return losses
