import numpy as np
import torch
import gym
import argparse
import os
import wandb
from utils import ReplayBuffer
from DQN.dqn import DQN
 
 
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval(policy, env_name, seed, eval_episodes=10):
    print(env_name)
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

def train(policy, env, replay_buffer, seed, max_timesteps, start_timesteps):
    # Evaluate untrained policy
    env_name = env.unwrapped.spec.id
    evaluations = [eval(policy, env_name, seed)]
    metrics = []

    state, done = env.reset(), False
    state = torch.tensor(state, dtype=torch.FloatTensor)
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
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Process data and store it in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = torch.tensor(next_state, dtype=torch.FloatTensor)
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            metrics.append(policy.train(*replay_buffer.sample(args.batch_size)))

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval(policy, env_name, seed))
            
    return metrics, evaluations


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="DQN")                        # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="LunarLander-v2")          	# OpenAI gym environment name
    parser.add_argument("--on_policy", action="store_true")
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)    # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)           # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)       # Max time steps to run environment
    parser.add_argument("--batch_size", default=256, type=int)          # Batch size for both actor and critic
    
    # include ?
    parser.add_argument("--discount", default=0.99)                     # Discount factor
    
    parser.add_argument("--save", action="store_true")                  # Save model and optimizer parameters
    parser.add_argument("--load", default="")                           # Model load file name, "" doesn't load, "default" uses file_name
    
    # model specific
    parser.add_argument("--tau", default=0.95, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    args = parser.parse_args()

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
    print(env.action_space.shape)
    state_dim = env.observation_space.shape[0]
    #action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.n)
    action_dim = int(max_action)
    print(action_dim, max_action)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "discount": args.discount,
        "tau": args.tau,
        "eps": args.eps
    }

    policy = DQN(**kwargs)

    if args.load != "":
        policy_file = file_name if args.load == "default" else args.load
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    train(
        policy, 
        env, 
        replay_buffer, 
        args.seed, 
        args.max_timesteps, 
        args.start_timesteps
    )
    
    
    
    
    