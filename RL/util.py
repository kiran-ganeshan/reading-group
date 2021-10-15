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
import wandb
from scipy.signal import lfilter
import jax
import flax
from flax import linen as jnn
from jax import numpy as jnp
from enum import Enum
from gym.spaces import Space, Box, Discrete

# types
from typing import Sequence, Callable, Any, Iterable, Dict, AnyStr, Optional
from jax.interpreters.xla import DeviceArray


def to_torch(tensor, dtype):
    return torch.tensor(tensor, dtype=dtype)  


def from_torch(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    else:
        return tensor
    
    
def one_hot(tensor, num_classes=-1):
    return F.one_hot(tensor, num_classes=num_classes).type(torch.float32)


def log_wandb(str, metrics, step=None):
    log_items = dict()
    for key, value in metrics.items():
        log_items[str + "_" + key] = from_torch(value)
    wandb.log(log_items, step=step)
    
    
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
    
    
def test_action_type(policy : nn.Module, state_dim : int):
    wrong_type = "select_action must return torch or numpy array"
    wrong_dtype = "select_action must return array with dtype int or float"
    assert policy.is_learner, "module must be learner to test action type"
    action = from_torch(policy.select_action(torch.zeros(state_dim)))
    assert isinstance(action, np.ndarray), wrong_type
    if np.issubdtype(action.dtype, np.integer):
        return ActionType.DISCRETE
    if np.issubdtype(action.dtype, np.floating):
        return ActionType.CONTINUOUS
    assert False, wrong_dtype


def learner(*modules, get_q=False):
    def transform(cls):
        prev_init = cls.__init__
        params = inspect.signature(prev_init).parameters
        parser = argparse.ArgumentParser(prog='model_init')

        bad_keys = ["self", "args", "kwargs"]
        required_keys = ["discount"]
        for name, param in params.items():
            if not name in bad_keys:
                if name in required_keys:
                    required_keys.remove(name)
                no_annotation = f"@module requires __init__ parameter {name} to be annotated with a"
                no_annotation += "type hint, e.g. x : int, y : bool, z : float"
                assert not (param.annotation is inspect.Parameter.empty), no_annotation
                
                positional_only = f"@module does not allow __init__ parameter {name} to be positional-only."
                assert not (param.kind is inspect.Parameter.POSITIONAL_ONLY), positional_only
                
                kwargs = {'type': param.annotation, 'required': True}
                if not (param.default is inspect.Parameter.empty):
                    kwargs['default'] = param.default
                    kwargs['required'] = False
                    
                parser.add_argument("--" + name, **kwargs)
        missing_keys = f"@module requires __init__ to have parameter discount"
        assert len(required_keys) == 0, missing_keys
                
        def init(self, args):
            args, extras = parser.parse_known_args(args)
            wrong_args = [arg[2:] for arg in extras if arg[0:2] == '--' and arg != '--discount']
            assert len(wrong_args) == 0, f"Supplied incorrect args {wrong_args}"
            
            prev_init(self, **vars(args))
            
            misnamed_mod = "Ensure all module python instance variables are named "
            misnamed_mod += "according to the inputs to learner."
            assert all([hasattr(self, mod) for mod in modules]), misnamed_mod
            
            self.modules = tuple([getattr(self, mod) for mod in modules])
            self.action_type = test_action_type(self, args.state_dim)
        
        cls.__init__ = init
        cls.is_learner = True    # For testing in training loops
        cls.get_q = get_q
        assert hasattr(cls, 'select_action'), "Must write a select_action function"
        assert hasattr(cls, 'train'), "Must write a training step"
        
        return cls
    return transform
        


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
    
    get_q               Whether to calculate Q-values. When True, trajectory_q and discount_qval control
                        how Q-values are calculated. Their defaults are set up so that we discount and sum
                        reward-to-go to obtain Q-values. trajectory_q and discount_qval allow a user to 
                        choose between computing Q-values by assigning Q(s, a) to the total reward
                        for the entire trajectory or just the reward-to-go. See below descriptions for
                        more detail.
                        
    
    trajectory_q        If True (and get_q is True), calculate Q-values across entire trajectories:
                        for all transitions (s, a) in the trajectory, Q(s, a) is the discounted total
                        reward across the entire trajectory. 
                        
                        Only affects anything if get_q is True (we are getting the Q-value).
                        
    discount_qval       If False (and get_q is True, and trajectory_q is False), calculate Q-values
                        by discounting and summing reward-to-go: for a transition (s_t, a_t) in the
                        trajectory, 
                        Q(s_t, a_t) <- sum[gamma^(t' - t) Q(s_t', a_t') from t'=t to t'=T]
    
                        If True (and get_q is True, and trajectory_q is False), calculate Q-values
                        using the remaining discounted reward in the trajectory: for a transition 
                        (s_t, a_t) in the trajectory, 
                        Q(s_t, a_t) <- sum[gamma^t' Q(s_t', a_t')) from t'=t to t'=T]
                        Q(s_t, a_t) <- gamma^t sum[gamma^(t' - t) Q(s_t', a_t') from t'=t to t'=T]
                        Notice this is gamma^t times the result for discount_qval=False, so this
                        argument controls whether we discount Q-values that appear later in trajectories.
                        
                        Only affects anything if get_q is True (we are getting the Q-value) and 
                        trajectory_q is False (we are not using whole-trajectory Q-values).
                        
    """
    
    def __init__(self, 
                 state_dim : int, 
                 action_dim : int, 
                 max_size : int = int(1e6), 
                 continuous : bool = False, 
                 get_q : bool = False,
                 trajectory_q : bool = False, 
                 discount_qval : bool = False,
                 discount : float = None,
                 lookahead_for_returns : int = None):
        self.max_size = max_size
        self.continuous = continuous
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.get_q = get_q
        self.trajectory_q = trajectory_q
        self.discount_qval = discount_qval
        self.discount = discount
        self.lookahead = lookahead_for_returns if lookahead_for_returns else 1
        self.output_lookahead = lookahead_for_returns is not None
        if get_q:
            assert discount, "Must provide discount to replay buffer if retrieving Q-value"

        self.reset()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        
        # Check types
        action_type = np.dtype('float32') if self.continuous else int
        inputs = [state, action, next_state, reward, done]
        expected_types = [np.dtype('float32'), action_type, np.dtype('float32'), np.float64, float]
        get_type = lambda x: (type(x) if not isinstance(x, np.ndarray) else x.dtype)
        types = list(map(get_type, inputs))
        assert types == expected_types, f"expected types {expected_types}\n but got types {types}"
        
        # Add data
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        
        # Get Q-values
        if self.get_q and done:
            self.get_qval()
            
        # Adjust pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        action_type = torch.float if self.continuous else torch.long
        qval = () if not self.get_q else (to_torch(self.qval[ind], dtype=torch.float32).to(self.device),)
        # Looks ahead the specified amount or until the end of the episode.
        lookahead = 0
        while lookahead < self.lookahead and self.not_done[(ind + lookahead) % self.max_size]:
            lookahead += 1
        # May need to wrap around the rewards subarray if the episode wraps around the end of the buffer.
        next_state = self.state[(ind + lookahead) % self.max_size]
        if ind + lookahead < self.max_size:
            rewards = self.reward[ind:ind+lookahead]
        else:
            rewards = self.reward[ind:self.max_size] + self.reward[0:ind+lookahead-self.max_size]
        nstep_return = lfilter([1], [1, -self.discount], rewards[::-1], axis=0)[-1]
        lookahead_tup = () if not self.output_lookahead else (lookahead,)
        return (
            to_torch(self.state[ind], dtype=torch.float).to(self.device),
            to_torch(self.action[ind], dtype=action_type).to(self.device),
            to_torch(next_state, dtype=torch.float).to(self.device),
            to_torch(nstep_return, dtype=torch.float).to(self.device),
            *qval,
            to_torch(self.not_done[ind], dtype=torch.float).to(self.device),
            *lookahead_tup
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
        if self.get_q:
            self.qval = np.zeros((self.max_size,), dtype=np.float32)
        self.not_done = np.zeros((self.max_size,), dtype=float)
        
    def get_qval(self):
        # Calculate trajectory indices and retrieve corresponding rewards
        traj_begin = self.ptr
        while self.not_done[(traj_begin - 1) % self.max_size]:
            traj_begin = (traj_begin - 1) % self.max_size
        if traj_begin < (self.ptr + 1) % self.max_size:
            first_idx = range(traj_begin, (self.ptr + 1) % self.max_size)
            second_idx = range(0, 0)
        else:
            assert self.size == self.max_size
            first_idx = range(traj_begin, self.max_size)
            second_idx = range(0, (self.ptr + 1) % self.max_size)
        traj_r = np.concatenate([self.reward[first_idx], self.reward[second_idx]])
            
        # Calculate Q-values
        if self.trajectory_q:
            traj_q = lfilter([1], [1, -self.discount], traj_r[::-1], axis=0)[-1]
            traj_q = np.repeat(traj_q, traj_r.shape[0])
        else:
            traj_q = lfilter([1], [1, -self.discount], traj_r[::-1], axis=0)[::-1]
            if self.discount_qval:
                discounts = np.array([pow(self.discount, i) for i in range(traj_r.shape[0])])
                traj_q = traj_q * discounts
                
        # Assign Q-values
        self.qval[first_idx] = traj_q[:len(first_idx)]
        self.qval[second_idx] = traj_q[len(first_idx):]


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