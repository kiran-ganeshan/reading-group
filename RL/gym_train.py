import numpy as np
import torch
import gym
from gym.spaces import Box, Discrete
import argparse
import os
import wandb
from tqdm import tqdm
import util
from algos.dqn.dqn import DQN
from algos.pg.pg import PG
from algos.ddpg.ddpg import DDPG
import collections
fields = ['state', 'action', 'reward', 'done', 'next_state']
Transition = collections.namedtuple('Transition', fields)

algos = {
    "DQN": DQN,
    "PG": PG,
    "DDPG": DDPG
}

envs = [
    'LunarLanderContinuous-v2'
]
 
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval(policy, env_name, seed, eval_episodes=10, render=False):
    assert env_name in envs
    eval_env = gym.make(env_name)
    if render:
        eval_env = util.VideoRecorder(eval_env)
    eval_env.seed(seed)

    rewards = []
    for _ in range(eval_episodes):
        state, done, ep_reward = eval_env.reset(), False, 0.
        while not done:
            input = util.to_torch(state, torch.float32)
            action = policy.select_action(input)
            action = util.from_torch(action)
            state, reward, done, _ = eval_env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    
    return {'mean_return': np.mean(rewards),
            'std_return': np.std(rewards),
            'max_return': np.max(rewards),
            'min_return': np.min(rewards)}

def train(policy, 
          env : gym.Env, 
          replay_buffer : util.ReplayBuffer, 
          on_policy : bool,
          seed : int, 
          batch_size : int,
          max_timesteps : int, 
          start_timesteps : int, 
          ep_len : int,
          train_freq : int,
          eval_freq : int,
          no_tqdm : bool = False, 
          save_logprob : bool = False):
    """
    Trains 
    """
    # Get env_name for evaluation environment creation
    env_name = env.unwrapped.spec.id

    # Prepare environment, state, and counters
    state, done = env.reset(), False
    episode_reward = 0.
    t = 0
    episode_num = 0

    for step in tqdm(range(int(max_timesteps)), disable=no_tqdm):
        t += 1

        # Select action randomly or according to policy
        if step < args.start_timesteps:
            action = env.action_space.sample()
        else:
            input = util.to_torch(state, torch.float32)
            result = policy.select_action(input, get_logprob=save_logprob)
            if save_logprob:
                action, logprob = result
                logprob = util.from_torch(logprob)
            else:
                action = result
            action = util.from_torch(action)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done = float(done) if t < ep_len else 0.

        # Process data and store it in replay buffer
        transition = (state, action, next_state, reward, done)
        if save_logprob:
            transition = (*transition, logprob)
        replay_buffer.add(*transition)
        episode_reward += reward
        state = next_state

        # Train agent after collecting sufficient data
        # Only trains if start_timesteps many steps have passed and either
        # - we are training off-policy and this is a training iteration (according to train_freq), or
        # - we are training on-policy and we are finished with an episode
        can_train = not on_policy and (step + 1) % train_freq == 0
        can_train = can_train or (on_policy and done)
        if step >= start_timesteps and can_train:
            data = replay_buffer.sample(batch_size)
            metrics = policy.train(*data)
            util.log_wandb('train', metrics, step=step + 1)

        # Complete episode if done
        if done: 
            episode_metrics = {'ep_len': t, 'ep_reward': episode_reward}
            util.log_wandb('train', episode_metrics, step=step + 1)
            
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0.
            t = 0
            episode_num += 1 
            
            # Dump the replay buffer if training on-policy
            if on_policy:
                replay_buffer.reset()
            
        # Evaluate episode
        if (step + 1) % eval_freq == 0:
            eval_metrics = eval(policy, env_name, seed)
            util.log_wandb('eval', eval_metrics, step=step + 1)
            
        # Save transition to offline dataset
        #if save_data:
            


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    trajectory_q_help = "Calculate whole-trajectory Q-value for each step -- only relevant when algo.get_q is True"
    discount_qval_help = "If true, discount qval calculations at mid-trajectory timesteps -- only relevant when \
                          algo.get_q is True (we are getting q-values) and trajectory_q is False (we are not using \
                          whole-trajectory q-values at each timestep)"
    
    # Environment and Training parameters
    parser.add_argument("--env", default="CartPole-v1", help="Gym env name, or \"<DMC Domain Name> <DMC Task Name>\"")
    parser.add_argument("--on_policy", "-on", action="store_true", help="Use single-trajectory buffer instead of replay buffer") 
    parser.add_argument("--eval_freq", default=int(5e3), type=int, help="Number of env steps between evaluations")
    parser.add_argument("--train_freq", default=10, type=int, help="Number of env steps between training")
    parser.add_argument("--start_timesteps", default=int(25e3), type=int, help="How long to use random policy") 
    parser.add_argument("--max_timesteps", default=int(1e6), type=int, help="Max number of env steps in training")
    parser.add_argument("--ep_len", "-T", default=int(1e3), type=int, help="Maximum episode length")  
    parser.add_argument("--seed", "-s", default=0, type=int, help="Sets Gym, PyTorch and Numpy seeds")
    
    # Model hyperparameters
    parser.add_argument("--algo", default="PG", help="Policy name (see gym_train.algos for possibilities)")   
    parser.add_argument("--batch_size", default=int(1e3), type=int, help="Batch size for replay buffer samples (if off policy)")
    parser.add_argument("--discount", default=0.97, type=float, help="Discount factor for future rewards")
    
    # Buffer hyperparameters
    parser.add_argument("--buffer_size", default=int(1e5), type=int, help="Size of Replay Buffer") 
    parser.add_argument("--trajectory_q", action="store_true", help=trajectory_q_help) 
    parser.add_argument("--discount_qval", action="store_true", help=discount_qval_help) 
    
    # I/O
    parser.add_argument("--save", action="store_true", help="Save model and optimizer parameter checkpoints in wandb")
    parser.add_argument("--load", type=str, help="Load model and optimizer params from this wandb run id")
    parser.add_argument("--wandb", action="store_true", help="Save training metrics, eval metrics, and data in wandb")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bar")
    
    args, model_args = parser.parse_known_args()

    file_name = f"{args.algo}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.algo}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    wandb.init(project='general_benchmarks', entity='mlab-rl-benchmarking')
    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    
    # Get environment dimensions
    state_dim, action_dim = util.get_env_dims(env)

    # Add environment dimensions and discount to model args
    model_args += ["--state_dim", str(state_dim)]
    model_args += ["--action_dim", str(action_dim)]
    model_args += ["--discount", str(args.discount)]

    assert algos[args.algo].is_learner, f"Ensure you call @learner on the {algos[args.algo]} class"
    policy = algos[args.algo](model_args)
    
    
    discrete_policy = util.policy_is_discrete(policy, state_dim)
    discrete_space = util.space_is_discrete(env.action_space)
    wrong_type = f"Policy is {'discrete' if discrete_policy else 'continuous'}\n"
    wrong_type += f"Space is {'discrete' if discrete_space else 'continuous'}"
    assert discrete_policy == discrete_space, wrong_type
    discrete = discrete_policy
    

    if args.load:
        policy.load(f"./models/{file_name}")

    replay_size = args.ep_len if args.on_policy else args.buffer_size
    batch_size = args.ep_len if args.on_policy else args.batch_size
    replay_buffer = util.ReplayBuffer(state_dim, 
                                      action_dim,
                                      max_size=replay_size, 
                                      discrete=discrete, 
                                      get_q=algos[args.algo].get_q,
                                      get_logprob = algos[args.algo].get_logprob,
                                      trajectory_q=args.trajectory_q, 
                                      discount_qval=args.discount_qval,
                                      discount=args.discount)
    
    train(
        policy, 
        env, 
        replay_buffer, 
        args.on_policy,
        args.seed, 
        batch_size,
        args.max_timesteps, 
        args.start_timesteps,
        args.ep_len,
        args.train_freq,
        args.eval_freq,
        args.no_tqdm
    )
    
    
    
    
    