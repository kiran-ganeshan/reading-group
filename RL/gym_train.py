import numpy as np
import torch
import gym
from gym.spaces import Box, Discrete
import argparse
import os
import wandb
from util import ReplayBuffer, ActionType, get_action_type
from DQN.dqn import DQN
from PG.pg import PG
algos = {
    "DQN": DQN,
    "PG": PG
}
 
 
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(torch.FloatTensor(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

def train(policy, 
          env : gym.Env, 
          replay_buffer : ReplayBuffer, 
          on_policy : bool,
          seed : int, 
          batch_size : int,
          max_timesteps : int, 
          start_timesteps : int, 
          ep_len : int,
          train_freq : int,
          eval_freq : int):
    """
    Trains 
    """
    # Evaluate untrained policy and prepare train metrics
    env_name = env.unwrapped.spec.id
    evaluations = [eval(policy, env_name, seed)]
    metrics = []

    # Prepare environment, state, and counters
    state, done = env.reset(), False
    state = torch.FloatTensor(state)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(state)

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < ep_len else 0

        # Process data and store it in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = torch.FloatTensor(next_state)
        episode_reward += reward

        # Train agent after collecting sufficient data
        # Only trains if 
        # - we are training off-policy and this is a training iteration (according to train_freq),
        # - we are training on-policy and we are finished with an episode
        if t >= start_timesteps and ((not on_policy and (t + 1) % train_freq == 0) or done):
            metrics = policy.train(*replay_buffer.sample(batch_size))

        # Complete episode if done
        if done: 
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            state = torch.FloatTensor(state)
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            # Reset the replay buffer if training on-policy
            if on_policy:
                replay_buffer.reset()
            
        # Evaluate episode
        if (t + 1) % eval_freq == 0:
            evaluations.append(eval(policy, env_name, seed))
            
    return metrics, evaluations


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # Algorithm, Environment, Training Style
    parser.add_argument("--algo", default="DQN")                        # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="LunarLander-v2")          	# OpenAI gym environment name
    parser.add_argument("--on_policy", "-on", action="store_true")      # Whether to train with on-policy (single trajectory)
                                                                        # or off-policy buffer
    # Hyperparameters
    parser.add_argument("--seed", "-s", default=0, type=int)            # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)    # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)           # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)       # Max time steps to run environment
    parser.add_argument("--ep_len", "-T", default=1e3, type=int)        # Maximum episode length
    parser.add_argument("--batch_size", default=256, type=int)          # Batch size for both actor and critic
    parser.add_argument("--buffer_size", default=1e5, type=int)         # Size of Replay Buffer
    
    # I/O
    parser.add_argument("--save", action="store_true")                  # Save model and optimizer parameters
    parser.add_argument("--load", action="store_true")                  # Load model and optimizer parameters
    
    args, model_args = parser.parse_known_args()

    file_name = f"{args.algo}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.algo}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_type = get_action_type(env.action_space)
    if action_type == ActionType.CONTINUOUS:
        action_dim = env.action_space.shape[0]
    else: 
        assert action_type == ActionType.DISCRETE
        action_dim = int(env.action_space.n)
        
    # Add environment dimensions and discount to model args
    model_args += ["--state_dim", str(state_dim)]
    model_args += ["--action_dim", str(action_dim)]

    policy = algos[args.algo](model_args)
    assert policy.action_type == action_type

    if args.load:
        policy.load(f"./models/{file_name}")

    replay_size = args.ep_len if args.on_policy else args.buffer_size
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=replay_size)
    
    train(
        policy, 
        env, 
        replay_buffer, 
        args.on_policy,
        args.batch_size,
        args.seed, 
        args.max_timesteps, 
        args.start_timesteps,
        args.ep_len
    )
    
    
    
    
    