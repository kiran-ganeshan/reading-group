# libraries
import numpy as np
import torch
from torch import TensorType
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import jax
import flax
from flax import linen as jnn
from jax import numpy as jnp

# types
from typing import Sequence, Callable, Any, Iterable, Dict, AnyStr
from jax.interpreters.xla import DeviceArray


class ReplayBuffer(object):
    
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


class MLP(nn.Module):
    def __init__(self, 
                 input_size : int, 
                 hidden_sizes : Sequence[int], 
                 output_size: int, 
                 activation : Callable[[TensorType], TensorType]):
        super(MLP, self).__init__()
        
        sizes = [input_size] + [x for x in hidden_sizes] + [output_size]
        self.layers = []
        for s1, s2 in zip(sizes[:-1], sizes[1:]):
            self.layers.append(nn.Linear(s1, s2))
        self.activation = activation
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x


class Module(jnn.Module):
    
    @property 
    def input_shape(self):
        pass

    
class JMLP(Module):

    input_size : int
    output_size : int
    hidden_sizes : Sequence[int]
    activation : Callable[[DeviceArray], DeviceArray] = lambda x: x
    
    @property
    def input_shape(self):
        return (self.input_size,)

    @jnn.compact
    def __call__(self, x):
        x = x.reshape((-1, self.input_size))
        return self.mlp(x)

    def mlp(self, x):
        for size in self.hidden_sizes:
            x = jnn.relu(jnn.Dense(features=size)(x))
        x = jnn.Dense(features=self.output_size)(x)
        x = self.activation(x)
        return x