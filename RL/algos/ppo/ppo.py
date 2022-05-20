import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import util
from scipy.signal import lfilter
    

@util.learner(get_logprob=True)
class PPO(object):
    def __init__(
        self,
        state_dim : int,
        action_dim : int,
        discount : float = 0.97,
        actor_lr : float = 1e-2,
        critic_lr : float = 1e-3,
        tau : float = 0.05,
        predict_std : bool = False,
        gae_lambda : float = None,
        advantage_n : int = None,
        epsilon : float = 0.1
    ):
        self.actor = util.GaussianPolicy(state_dim=state_dim, 
                                         action_dim=action_dim, 
                                         hidden_sizes=(256, 256), 
                                         activation=nn.ReLU(),
                                         final_activation=nn.LogSoftmax(dim=-1),
                                         predict_std=predict_std)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        self.critic = util.MLP(input_size=state_dim,
                               output_size=1,
                               hidden_sizes=(256, 256),
                               activation=nn.ReLU(),
                               final_activation=nn.Identity())
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.discount = discount
        self.tau = tau
        self.action_dim = action_dim
        self.gae_lambda = gae_lambda
        self.adv_n = advantage_n
        self.epsilon = epsilon

    def select_action(self, state, deterministic=True, get_logprob=False):
        action = self.actor(state, deterministic=deterministic)
        if get_logprob:
            logprob =  self.actor.log_prob(state, action)
            return action, logprob
        else:
            return action
    
    def estimate_adv(self, td_error, not_done):
        if self.gae_lambda:
            decay = self.discount * self.gae_lambda
            adv = lfilter([1], [1, -decay], (not_done * td_error)[::-1])[::-1]
            adv += (1. - not_done) * td_error
        elif self.adv_n:
            adv = (not_done * td_error)[self.adv_n - 1:]
            adv = pow(self.discount, self.adv_n) * torch.concat(adv, torch.zeros(self.adv_n - 1))
            adv = lfilter([1], [1, -self.discount], (not_done * td_error - adv)[::-1])[::-1]
            adv += (1. - not_done) * td_error
        else:
            adv = td_error
        return adv

    def train(self, *data):
        state, action, next_state, reward, not_done, old_logprob = data

        # Compute the target V value
        with torch.no_grad(): 
            target_V = self.critic_target(next_state)
            target_V = reward + self.discount * not_done * target_V
            
        # Get current V estimates
        V = self.critic(state)
 
        # Compute critic loss (and TD-error for surrogate objective)
        td_error = target_V - V
        critic_loss = F.mse_loss(td_error, torch.zeros_like(td_error))
 
        # Compute surrogate objective
        adv = self.estimate_adv(td_error.detach(), not_done)
        logprob = self.actor.log_prob(state, action)
        imw = torch.exp(logprob - old_logprob)     # importance weight
        clipped_imw = torch.clip(imw, 1. - self.epsilon, 1. + self.epsilon)
        actor_loss = -torch.mean(torch.min(imw * adv, clipped_imw * adv), dim=0)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {'actor_loss': actor_loss, 
                'critic_loss': critic_loss}