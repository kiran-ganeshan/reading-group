import copy
import torch
import torch.nn as nn
from torch.functional import F
import util
    

@util.learner()
class DDPG(object):
    def __init__(
        self,
        state_dim : int,
        action_dim : int,
        discount : float = 0.97,
        actor_lr : float = 1e-3,
        critic_lr : float = 1e-3,
        actor_ema : float = 0.05,
        critic_ema : float = 0.05
    ):
        self.critic = util.MLP(input_size=state_dim + action_dim, 
                               output_size=1, 
                               hidden_sizes=(256, 256), 
                               activation=nn.ReLU(),
                               final_activation=nn.Identity())
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.actor = util.MLP(input_size=state_dim, 
                              output_size=action_dim, 
                              hidden_sizes=(256, 256), 
                              activation=nn.ReLU(),
                              final_activation=nn.LogSoftmax(dim=-1))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.discount = discount
        self.actor_ema = actor_ema
        self.critic_ema = critic_ema
        self.action_dim = action_dim

    def select_action(self, state, deterministic=True):
        action = self.actor(state)
        if not deterministic:
            action += torch.rand_like(action) * self.eps
        return action

    def train(self, *data):
        state, action, next_state, reward, not_done = data
        
        # Compute the target Q value
        with torch.no_grad(): 
            next_action = self.actor_target(next_state)
            next_Q = self.critic_target(next_state, next_action)
            target_Q = reward + self.discount * not_done * next_Q
            
        # Compute critic loss and optimize critic
        Q = self.critic(state, action)
        critic_loss = F.mse_loss(Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
 
        # Compute actor surrogate objective and optimize actor
        actor_loss = -torch.mean(self.critic(state, self.actor(state)), dim=0)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for nn, target, ema in [(self.critic, self.critic_target, self.actor_ema), 
                                (self.actor, self.actor_target, self.critic_ema)]:
            for param, target_param in zip(nn.parameters(), target.parameters()):
                target_param.data.copy_(ema * param.data + (1 - ema) * target_param.data)
                
        return {'actor_loss': actor_loss,
                'critic_loss': critic_loss}