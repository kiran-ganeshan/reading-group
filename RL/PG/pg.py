import copy
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import util
    

@util.learner(get_q=True)
class PG(object):
    def __init__(
        self,
        state_dim : int,
        action_dim : int,
        discount : float = 0.97,
        lr : float = 1e-2,
        tau : float = 0.05,
        #eps : float = 1e-8
    ):
        self.actor = util.MLP(input_size=state_dim, 
                              output_size=action_dim, 
                              hidden_sizes=(256, 256), 
                              activation=nn.ReLU(),
                              final_activation=nn.LogSoftmax(dim=-1))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.discount = discount
        self.tau = tau
        self.eps = eps
        self.action_dim = action_dim

    def select_action(self, state, deterministic=True):
        if deterministic:
            return torch.argmax(self.actor(state), -1)
        else:
            return Categorical(logits=self.actor(state)).sample()

    def train(self, *data):
        state, action, next_state, reward, qval, not_done = data

        # Get current policy
        policy = self.actor(state)
 
        # Compute surrogate objective
        mask = util.one_hot(action, num_classes=self.action_dim)
        logprob = torch.sum(policy * mask, dim=-1)
        loss = -torch.mean(qval * logprob, dim=0)
        losses = {'loss': loss}

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        
        return losses