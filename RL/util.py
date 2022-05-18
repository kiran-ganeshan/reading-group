# libraries
import inspect
import argparse
import numpy as np
import torch
from torch import TensorType
import torch.nn as nn
import torch.nn.functional as F
import wandb
from scipy.signal import lfilter
from enum import Enum
from gym.spaces import Space, Box, Discrete
import os
import gym
# types
from typing import Sequence, Callable


device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def to_torch(tensor, dtype=torch.float32, *args, **kwargs):
    return torch.from_numpy(tensor, *args, **kwargs).type(dtype).to(device)


def from_torch(tensor, *args, **kwargs):
    if isinstance(tensor, torch.Tensor):
        return tensor.to('cpu').detach().numpy()
    else:
        return tensor
    
    
def one_hot(tensor, num_classes=-1):
    return F.one_hot(tensor, num_classes=num_classes).type(torch.float32)


def log_wandb(str, metrics, step=None):
    log_items = dict()
    for key, value in metrics.items():
        log_items[str + "_" + key] = from_torch(value)
    wandb.log(log_items, step=step)
    
    
def space_is_discrete(action_space : Space):
    return isinstance(action_space, Discrete)
    
    
def policy_is_discrete(policy, state_dim : int):
    wrong_type = "select_action must return torch array"
    assert policy.is_learner, "module must be learner to test action type"
    action = policy.select_action(torch.zeros(state_dim))
    assert isinstance(action, torch.Tensor), wrong_type
    return not torch.is_floating_point(action)


def get_env_dims(env: gym.Env):
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, Discrete): 
        action_dim = int(env.action_space.n)
    return state_dim, action_dim


def learner(get_q=False, get_logprob=False):
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
            
            
        
        cls.__init__ = init
        cls.is_learner = True    # For testing in training loops
        cls.get_q = get_q
        cls.get_logprob = get_logprob
        assert hasattr(cls, 'select_action'), "Must write a policy in function select_action"
        assert hasattr(cls, 'train'), "Must write a training step in function select_action"
        
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
    
    get_q               Whether to calculate Monte-Carlo estimates of Q-values. When True, trajectory_q and 
                        discount_qval control how Q-value estimates are calculated. Their defaults are set up 
                        so that we discount and sum reward-to-go to obtain Q-values.
                        
    get_logprob         Whether to save the log-probability of the action under the current policy when saving 
                        a transition.
    
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
                 discrete : bool = True, 
                 get_q : bool = False,
                 get_logprob : bool = False,
                 trajectory_q : bool = False, 
                 discount_qval : bool = False,
                 discount : float = None):
        self.max_size = max_size
        self.discrete = discrete
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.get_q = get_q
        self.get_logprob = get_logprob
        self.trajectory_q = trajectory_q
        self.discount_qval = discount_qval
        self.discount = discount
        
        if get_q:
            assert discount, "Must provide discount to replay buffer if retrieving Q-value"

        self.reset()

    def add(self, 
            state : np.ndarray, 
            action : np.ndarray, 
            next_state : np.ndarray, 
            reward : np.ndarray, 
            done : np.ndarray,
            logprob : np.ndarray = None):
        
        # Ensure we have logprob if necessary
        if self.get_logprob:
            assert logprob
        
        # Change types
        state = state.astype('float')
        if not self.discrete:
            action = action.astype('float')
        next_state = next_state.astype('float')
        
        # Add data
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        if self.get_logprob:
            self.logprob[self.ptr] = logprob
        
        # Get Q-values
        if self.get_q and done:
            self.get_qval()
            
        # Adjust pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        action_type = torch.float if self.continuous else torch.long
        qval = () if not self.get_q else (to_torch(self.qval[ind], dtype=torch.float),)
        logprob = () if not self.get_logprob else (to_torch(self.logprob[ind], dtype=torch.float32),)
        return (
            to_torch(self.state[ind], dtype=torch.float),
            to_torch(self.action[ind], dtype=action_type),
            to_torch(self.next_state[ind], dtype=torch.float),
            to_torch(self.reward[ind], dtype=torch.float),
            *qval,
            to_torch(self.not_done[ind], dtype=torch.float),
            *logprob
        )
        
    def reset(self):
        self.ptr = 0
        self.size = 0
        
        action_type = np.int32 if self.discrete else np.float32
        action_size = (self.max_size,) + (() if self.discrete else (self.action_dim,))
        
        self.state = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.action = np.zeros(action_size, dtype=action_type)
        self.next_state = np.zeros((self.max_size, self.state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size,), dtype=np.float32)
        if self.get_q:
            self.qval = np.zeros((self.max_size,), dtype=np.float32)
        self.not_done = np.zeros((self.max_size,), dtype=float)
        if self.get_logprob:
            self.logprob = np.zeros((self.max_size,), dtype=np.float32)
        
    def get_qval(self):
        # Calculate trajectory indices and retrieve corresponding rewards
        assert self.get_q
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
    
class GaussianPolicy(MLP):
    
    def __init__(self, 
                 state_dim : int, 
                 hidden_sizes : Sequence[int], 
                 action_dim: int, 
                 activation : Callable[[TensorType], TensorType] = None,
                 final_activation : Callable[[TensorType], TensorType] = None,
                 predict_std : bool = False):
        output_size = (2 if predict_std else 1) * action_dim
        super(GaussianPolicy, self).__init__(state_dim, hidden_sizes, output_size, activation, final_activation)
        if not predict_std:
            self.log_std = nn.Parameter(torch.zeros((action_dim,)))
        self.predict_std = predict_std
        
    def _get_dist_params(self, state):
        out = super(GaussianPolicy, self).forward(state)
        if self.predict_std:
            mu, log_sigma = torch.split(out, 2)
        else:
            mu = out
            log_sigma = self.log_std
        return mu, log_sigma
        
    def forward(self, state, deterministic=False):
        mu, log_sigma = self._get_dist_params(state)
        delta = torch.rand_like(log_sigma) if not deterministic else 0.
        return mu + delta * torch.exp(log_sigma)
    
    def log_prob(self, state, action):
        mu, log_sigma = self._get_dist_params(state)
        log_prob = -torch.sum(log_sigma, -1) 
        log_prob -= 1/2 * torch.norm((action - mu) / torch.exp(log_sigma)) ** 2
        log_prob -= mu.shape[-1] * torch.log(2 * torch.pi) / 2
        return log_prob
        
        

'''
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
'''




# Take from
# https://github.com/denisyarats/pytorch_sac/
class VideoRecorder(gym.Wrapper):
    def __init__(self,
                 env: gym.Env,
                 height: int = 128,
                 width: int = 128,
                 fps: int = 30):
        super().__init__(env)

        self.current_episode = 0
        self.height = height
        self.width = width
        self.fps = fps
        self.frames = []

    def step(self, action: np.ndarray):

        frame = self.env.render(mode='rgb_array')

        if frame is None:
            try:
                frame = self.sim.render(width=self.width,
                                        height=self.height,
                                        mode='offscreen')
                frame = np.flipud(frame)
            except:
                raise NotImplementedError('Rendering is not implemented.')

        self.frames.append(frame)

        observation, reward, done, info = self.env.step(action)

        if done:
            wandb.log({'eval_video': wandb.Video(np.array(self.frames), fps=self.fps)}, commit=False)
            self.frames = []
            self.current_episode += 1

        return observation, reward, done, info