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
        lr : float = 1e-3,
        actor_ema : float = 0.05,
        critic_ema : float = 0.05,
        init_alpha : float = 1.,
        entropy_limit : float = 1.,
    ):
        self.critic = util.MLP(input_size=state_dim, 
                               output_size=action_dim, 
                               hidden_sizes=(256, 256), 
                               activation=nn.ReLU(),
                               final_activation=nn.Identity())
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        self.actor = util.GaussianPolicy(state_dim=state_dim, 
                                         action_dim=action_dim, 
                                         hidden_sizes=(256, 256), 
                                         activation=nn.ReLU(),
                                         final_activation=nn.LogSoftmax(dim=-1))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.alpha_optimizer = torch.optim.SGD(self.alpha)

        self.discount = discount
        self.actor_ema = actor_ema
        self.critic_ema = critic_ema
        self.init_alpha = init_alpha
        self.minent = entropy_limit

    def select_action(self, state, deterministic=True):
        return self.actor(state, deterministic=deterministic)

    def train(self, *data):
        state, action, next_state, reward, not_done = data
        
        # Compute the target Q value
        with torch.no_grad(): 
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q -= self.alpha * self.actor.log_prob(state, next_action)
            data = {'next_q': target_Q}
            target_Q = reward + self.discount * not_done * target_Q
            data = {'target_q': target_Q, **data}
 
        # Compute critic loss and optimize critic
        Q = self.critic(state, action)
        critic_loss = F.mse_loss(Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
 
        # Compute actor surrogate objective and optimize actor
        actor_loss = -torch.mean(self.critic(state, self.actor(state)), dim=0)
        actor_loss += self.alpha * self.actor.log_prob(state, self.actor(state))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha for entropy constraint
        alpha_loss = -self.alpha * (self.minent + self.actor.log_prob(state, action))
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        
        losses = {'actor_loss': actor_loss, 'critic_loss': critic_loss, 'temperature_loss': alpha_loss}

        # Update the frozen target models
        for nn, target, ema in [(self.critic, self.critic_target, self.actor_ema), 
                                (self.actor, self.actor_target, self.critic_ema)]:
            for param, target_param in zip(nn.parameters(), target.parameters()):
                target_param.data.copy_(ema * param.data + (1 - ema) * target_param.data)
        return losses