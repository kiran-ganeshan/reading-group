# libraries
from abc import abstractmethod
import inspect
import argparse
from itertools import islice
from collections import OrderedDict
import numpy as np
import torch
from torch import TensorType
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from scipy.signal import lfilter
import jax
import flax
from flax import linen as jnn
from jax import numpy as jnp
from enum import Enum
from gym.spaces import Space, Box, Discrete

# types
from typing import Sequence, Callable, Any, Iterable, Dict, AnyStr
from jax.interpreters.xla import DeviceArray



def module(cls):
    prev_init = cls.__init__
    params = inspect.signature(cls.__init__).parameters
    parser = argparse.ArgumentParser(prog='model_init')
    for name, param in params.items():
        if name != 'self':
            no_annotation = "@module requires all __init__ parameters except for self "
            no_annotation += "to be annotated with type hints, e.g. x : int, y : bool, z : float"
            assert not (param.annotation is inspect.Parameter.empty), no_annotation
            positional_only = "@module does not allow __init__ parameters to be positional-only."
            assert not (param.kind is inspect.Parameter.POSITIONAL_ONLY), positional_only
            kwargs = {'type': param.annotation, 'required': True}
            if not (param.default is inspect.Parameter.empty):
                kwargs['default'] = param.default
                kwargs['required'] = False
            parser.add_argument("--" + name, **kwargs)
    def init(self, args):
        print(args)
        args = parser.parse_args(args)
        print(vars(args))
        return prev_init(self, **vars(args))
    cls.__init__ = init
    return cls



class ActionType(Enum):
    DISCRETE = 0
    CONTINUOUS = 1
    BOTH = 2
    
    
    
def get_action_type(action_space : Space):
    if isinstance(action_space, Box):
        return ActionType.CONTINUOUS
    elif isinstance(action_space, Discrete):
        return ActionType.DISCRETE
    else:
        pass
        


class ReplayBuffer(object):
    """
    A Replay Buffer for RL Algorithms. Can adapt to continuous or discrete adtion spaces, can calculate
    Monte-Carlo Q-Value estimates (in two ways, standard Q-value calculation and discounted rewards to go).
    Arguments:
    state_dim           The number of dimensions in a continuous state space.
    action_dim          The number of dimensions in a continuous action space, or the maximum action in
                        a discrete action space. (Assumes action space is Discrete or Box)
    max_size            The maximum size of the replay buffer.
    continuous          Whether the action space is continuous.
    rewards_to_go       Whether to apply the discount on the rewards-to-go or the whole trajectory.
    """
    
    def __init__(self, state_dim, action_dim, max_size=int(1e6), continuous=False, rewards_to_go=False):
        self.max_size = max_size
        self.continuous = continuous
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rtg = rewards_to_go

        self.reset()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done, t):
        action_type = np.float32 if self.continuous else np.int32
        inputs = [state, action, next_state, reward, done, t]
        expected_types = [np.float32, action_type, np.float32, np.float32, np.int8, np.int32]
        assert list(map(lambda x: x.dtype, inputs)) == expected_types
        
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.t[self.ptr] = t

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, discount, get_t=True):
        ind = np.random.randint(0, self.size, size=batch_size)
        action_transform = torch.FloatTensor if self.continuous else torch.LongTensor
        t = (torch.IntTensor(self.t[ind]),) if get_t else ()
        qval = () if discount is None else (self.get_qval(discount),)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            action_transform(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            *qval,
            *t,
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
        
    def reset(self):
        self.ptr = 0
        self.size = 0
        
        action_type = np.float32 if self.continuous else np.int32
        action_size = (self.max_size,) + ((self.action_dim,) if self.continuous else ())
        
        self.state = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.action = np.zeros(action_size, dtype=action_type)
        self.next_state = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size,), dtype=np.float32)
        self.not_done = np.zeros((self.max_size,), dtype=np.int8)
        self.t = np.zeros((self.max_size,), dtype=np.int32)
        
    def get_qval(self, gamma):
        if self.rtg:
            rtg = self.reward * np.array([pow(gamma, t) for t in self.t])
            rtg = lfilter([1, -1], [1], rtg[::-1], axis=0)[::-1]
            return rtg
        else:
            qval = self.reward * self.not_done
            qval = lfilter([1, -gamma], [1], qval[::-1], axis=0)[::-1]
            return qval
            


class MLP(nn.Module):
    
    def __init__(self, 
                 input_size : int, 
                 hidden_sizes : Sequence[int], 
                 output_size: int, 
                 activation : Callable[[TensorType], TensorType] = None,
                 final_activation : Callable[[TensorType], TensorType] = None):
        super(MLP, self).__init__()
        
        sizes = [input_size] + [x for x in hidden_sizes] + [output_size]
        self.layers = []
        for i, (s1, s2) in enumerate(zip(sizes[:-1], sizes[1:])):
            layer = nn.Linear(s1, s2)
            self.layers.append(layer)
            self.add_module(f'layer{i}', layer)
        self.activation = activation if activation else nn.Identity()
        self.final_activation = final_activation if final_activation else activation
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.activation(x)
        x = self.final_activation(x)
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