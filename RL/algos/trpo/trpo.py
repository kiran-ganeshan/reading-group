import torch
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import util
from scipy.signal import lfilter

class ConjugateGradientOptimizer(Optimizer):
    
    def __init__(self, params, lr, cg_iters=10, bt_iters=15, max_constraint=1.,
                 bt_decay=0.8, reg_coeff=1e-5, tol=1e-10):
        defaults = {'reg_coeff': reg_coeff, 'tol': tol, 'lr': lr, 'max_constraint': max_constraint,
                    'cg_iters': cg_iters, 'bt_iters': bt_iters, 'bt_decay': bt_decay}
        super(ConjugateGradientOptimizer, self).__init__(params, defaults)
        self.reg_coeff = reg_coeff
        self.tol = tol
        self.lr = lr
        self.max_constraint = max_constraint
        self.cg_iters = cg_iters
        self.bt_decay = bt_decay
        self.bt_iters = bt_iters
        
        # Create helper functions for flattening/unflatting params and grads
        params, _ = self._get_params_and_grads()
        shapes = [p.shape or torch.Size([1]) for p in params]
        sizes = [np.prod(sh) for sh in shapes]
        idxs = np.cumsum(sizes)[:-1]
        self.unflat = lambda v: [p.reshape(sh) for p, sh in zip(np.split(v, idxs), shapes)]
        self.flat = lambda v: torch.cat([h.reshape(-1) for h in v])
        
    def _get_params_and_grads(self):
        params = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p)
                g = p.grad
                if g is None:
                    g = torch.zeros_like(p) # set all None grads to 0
                grads.append(g.reshape(-1))
        return params, grads
    
    def _backtrack(self, params, step, f_loss, f_constraint):
        prev_loss = f_loss()
        alpha = self.lr
        old_params = self.flat(params)
        for _ in range(self.bt_iters):
            new_params = self.unflat(old_params - alpha * step)
            for param, new_param in zip(params, new_params):
                param.data.copy_(new_param.data)
            if f_loss() < prev_loss and f_constraint() < self.max_constraint:
                break
            alpha = alpha * self.bt_decay
            
    def _create_hessian_func(self, params, f_constraint):
        const_grad = torch.autograd.grad(f_constraint(), params, create_graph=True)
        assert len(const_grad) == len(params)
        def H(v):
            unflat_v = self.unflat(v)
            gvp = torch.sum(torch.stack([torch.sum(g * x) for g, x in zip(const_grad, unflat_v)]))
            hvp = list(torch.autograd.grad(gvp, params, retain_graph=True))
            for i, (hx, p) in enumerate(zip(hvp, params)):
                if hx is None:
                    hvp[i] = torch.zeros_like(p)
            flat_hvp = self.flat(hvp)
            return flat_hvp + self.reg_coeff * v
        return H
            
    def _conjugate_grad(self, grads, H):
        grads = self.flat(grads)
        p = grads.clone()
        res = grads.clone()
        x = torch.zeros_like(grads)
        error = torch.dot(res, res)
        for _ in range(self.cg_iters):
            z = H(p)
            u = error / torch.dot(p, z)
            x += u * p
            res -= u * z
            new_error = torch.dot(res, res)
            p = res + (new_error / error) * p
            error = new_error
            if error < self.tol:
                break
        x += self.reg_coeff * grads
        return np.sqrt(2.0 * self.max_constraint / grads @ x) * x
        
    #@torch.no_grad()
    def step(self, f_loss, f_constraint):
        params, grads = self._get_params_and_grads()
        H = self._create_hessian_func(params, f_constraint)
        step = self._conjugate_grad(grads, H)
        self._backtrack(params, step, f_loss, f_constraint)
    

@util.learner(get_logprob=True)
class TRPO(object):
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
        cg_iters : int = 10, 
        bt_iters : int = 15, 
        max_constraint : float = 1.,
        bt_decay : float = 0.8, 
        reg_coeff : float = 1e-5, 
        tol : float = 1e-10
    ):
        self.actor = util.GaussianPolicy(state_dim=state_dim, 
                                         action_dim=action_dim, 
                                         hidden_sizes=(256, 256), 
                                         activation=nn.ReLU(),
                                         final_activation=nn.LogSoftmax(dim=-1),
                                         predict_std=predict_std)
        self.actor_optimizer = ConjugateGradientOptimizer(self.actor.parameters(), actor_lr, 
                                                          cg_iters, bt_iters, max_constraint,
                                                          bt_decay, reg_coeff, tol)
        
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
 
        # Define surrogate objective and constraint
        adv = self.estimate_adv(td_error.detach(), not_done)
        def actor_loss():
            self.actor_optimizer.zero_grad()
            logprob = self.actor.log_prob(state, action)
            imw = torch.exp(logprob - old_logprob)     # importance weight
            return -torch.mean(imw * adv, dim=0)
        def kl_constraint():
            logprob = self.actor.log_prob(state, action)
            return torch.mean(old_logprob - logprob)

        # Optimize the actor
        self.actor_optimizer.step(actor_loss, kl_constraint)
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {'critic_loss': critic_loss,
                'actor_loss': actor_loss()}