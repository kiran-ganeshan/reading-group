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


class Learner(object):
    
    def __init__(modules : Dict[nn.Module],
                 optimizers : Dict[Optimizer],
                 )



class MLP(nn.Module):
    
    def __init__(self, 
                 input_size : int, 
                 hidden_sizes : Sequence[int], 
                 output_size: int, 
                 activation : Callable[[TensorType], TensorType]):
        sizes = [input_size] + [x for x in hidden_sizes] + [output_size]
        self.layers = []
        for s1, s2 in zip(sizes[:-1], sizes[1:]):
            self.layers.append(nn.Linear(s1, s2))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x


class Module(jnn.Module):
    
    @property 
    def input_shape(self):
        pass

    
class MLP(Module):

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