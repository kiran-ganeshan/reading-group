import copy
import torch
import torch.nn as nn
from torch.functional import F
import util
import itertools
    

@util.learner()
class TD3(object):
    def __init__(
        self,
        state_dim : int,
        action_dim : int,
        discount : float = 0.97,
        lr : float = 1e-3,
        actor_ema : float = 0.05,
        critic_ema : float = 0.05,
        action_noise_std : float = 0.5,
        target_noise_std : float = 0.5,
        target_noise_limit : float = 1.
    ):
        self.critics = [util.MLP(input_size=state_dim + action_dim, 
                                 output_size=1, 
                                 hidden_sizes=(256, 256), 
                                 activation=nn.ReLU(),
                                 final_activation=nn.Identity()) for _ in range(2)]
        self.critic_targets = [copy.deepcopy(critic) for critic in self.critics]
        params = itertools.chain(*[mod.parameters() for mod in self.critics])
        self.critic_optimizer = torch.optim.Adam(params, lr=lr)
        
        self.actor = util.MLP(input_size=state_dim, 
                              output_size=action_dim, 
                              hidden_sizes=(256, 256), 
                              activation=nn.ReLU(),
                              final_activation=nn.LogSoftmax(dim=-1))
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.discount = discount
        self.actor_ema = actor_ema
        self.critic_ema = critic_ema
        self.action_dim = action_dim
        self.test_noise = action_noise_std
        self.train_noise = target_noise_std
        self.noise_clip = target_noise_limit

    def select_action(self, state, deterministic=True):
        action = self.actor(state)
        if not deterministic:
            action += self.test_noise * torch.rand_like(action) 
        return action

    def train(self, *data):
        state, action, next_state, reward, not_done = data
        
        # Compute the target Q value
        with torch.no_grad(): 
            action_noise = self.train_noise * torch.rand_like(action)
            action_noise = torch.clip(action_noise, -self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_state) + action_noise
            target_Qs = [ct(next_state, next_action) for ct in self.critic_targets]
            target_Q = torch.minimum(*target_Qs)
            target_Q = reward + self.discount * not_done * target_Q
 
        # Compute critic loss and optimize critic
        critic_loss = sum([F.mse_loss(c(state, action), target_Q) for c in self.critics])
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
 
        # Compute actor surrogate objective and optimize actor
        actor_loss = -torch.mean(self.critics[0](state, self.actor(state)), dim=0)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for nn, target, ema in zip(self.critics + [self.actor], 
                                   self.critic_targets + [self.actor_target], 
                                   2 * [self.critic_ema] + [self.actor_ema]):
            for param, target_param in zip(nn.parameters(), target.parameters()):
                target_param.data.copy_(ema * param.data + (1 - ema) * target_param.data)
                
        return {'actor_loss': actor_loss, 
                'critic_loss': critic_loss}