import copy
import torch
import torch.nn as nn
from torch.functional import F
import util
    

@util.learner
class DDPG(object):
    def __init__(
        self,
        state_dim : int,
        action_dim : int,
        discount : float = 0.97,
        lr : float = 1e-3,
        tau : float = 0.05,
        eps : float = 1e-2
    ):
        self.critic = util.MLP(input_size=state_dim, 
                               output_size=action_dim, 
                               hidden_sizes=(256, 256), 
                               activation=nn.ReLU(),
                               final_activation=nn.Identity())
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.actor = util.MLP(input_size=state_dim, 
                              output_size=action_dim, 
                              hidden_sizes=(256, 256), 
                              activation=nn.ReLU(),
                              final_activation=nn.LogSoftmax(dim=-1))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.discount = discount
        self.tau = tau
        self.eps = eps
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
            target_Q = self.critic_target(next_state)
            target_Q = torch.max(target_Q, -1)[0]
            data = {'next_q': target_Q}
            target_Q = reward + self.discount * not_done * target_Q
            data = {'target_q': target_Q, **data}
            
        # Get current Q estimates
        q_mask = util.one_hot(action, num_classes=self.action_dim)
        Q = torch.sum(self.critic(state) * q_mask, dim=-1)
 
        # Compute critic loss
        critic_loss = F.mse_loss(Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Get current policy
        policy = self.actor(state)
 
        # Compute actor surrogate objective
        mask = util.one_hot(action, num_classes=self.action_dim)
        logprob = torch.sum(policy * mask, dim=-1)
        actor_loss = -torch.mean(Q * logprob, dim=0)
        
        losses = {'actor_loss': actor_loss, 'critic_loss': critic_loss}
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        return losses