import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import MLP


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.mlp = MLP(input_size=state_dim, 
                       ouptut_size=action_dim, 
                       hidden_sizes=(256, 256), 
                       activation=nn.ReLU)


    def forward(self, state, action):
        return self.mlp(torch.cat([state, action], 1))
    

class DQN(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        discount=0.99,
        tau=0.005
    ):

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.action_dim = action_dim

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, *data):
        self.total_it += 1
        state, action, next_state, reward, not_done = data

        with torch.no_grad():
            

            # Compute the target Q value
            target_Q = self.critic_target(next_state)
            target_Q = torch.max(target_Q, -1)
            data = {'next_q': target_Q}
            target_Q = reward + not_done * self.discount * target_Q
            data = {'target_q': target_Q, **data}

        # Get current Q estimates
        action = F.one_hot(action, num_classes=self.action_dim)
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        losses = {'critic_loss': critic_loss}

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Q = self.critic(state, pi)
            lmbda = self.alpha/Q.abs().mean().detach()

            actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action) 
            losses = {'actor_loss': actor_loss, **losses}
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
        losses, data


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)