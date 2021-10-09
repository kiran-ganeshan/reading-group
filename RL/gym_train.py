import numpy as np
import torch
import gym
from gym.spaces import Box, Discrete
import argparse
import os
import wandb
from util import ReplayBuffer, ActionType, get_action_type
from util import log_wandb, to_torch, from_torch
from DQN.dqn import DQN
from PG.pg import PG
from DeulingDQN.deuling_dqn import DeulingDQN
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
            action = from_torch(policy.select_action(to_torch(state, torch.float32)))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return {'return': avg_reward}

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
    # Get env_name for evaluation environment creation
    env_name = env.unwrapped.spec.id

    # Prepare environment, state, and counters
    state, done = env.reset(), False
    episode_reward = 0.
    t = 0
    episode_num = 0

    for step in range(int(max_timesteps)):
        t += 1

        # Select action randomly or according to policy
        if step < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = from_torch(policy.select_action(to_torch(state, torch.float32)))

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if t < ep_len else 0.
        reward = np.float64(reward)

        # Process data and store it in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)
        episode_reward += reward
        state = next_state

        # Train agent after collecting sufficient data
        # Only trains if start_timesteps many steps have passed and either
        # - we are training off-policy and this is a training iteration (according to train_freq), or
        # - we are training on-policy and we are finished with an episode
        if step >= start_timesteps and ((not on_policy and (step + 1) % train_freq == 0) or done):
            metrics = policy.train(*replay_buffer.sample(batch_size))
            log_wandb('train', metrics, step=step + 1)

        # Complete episode if done
        if done: 
            print(f"Total T: {step+1} Episode Num: {episode_num+1} Episode T: {t} Reward: {episode_reward:.3f}")
            
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0.
            t = 0
            episode_num += 1 
            
            # Reset the replay buffer if training on-policy
            if on_policy:
                replay_buffer.reset()
            
        # Evaluate episode
        if (step + 1) % eval_freq == 0:
            eval_metrics = eval(policy, env_name, seed)
            log_wandb('eval', eval_metrics, step=step + 1)


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
    
    args, model_args = parser.parse_known_args()

    file_name = f"{args.algo}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.algo}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save and not os.path.exists("./models"):
        os.makedirs("./models")

    wandb.init(project='general_benchmarks', entity='mlab-rl-benchmarking')
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
    model_args += ["--discount", str(args.discount)]

    policy = algos[args.algo](model_args)
    wandb.watch(policy.modules)
    
    wrong_type = f"Using {algos[args.algo]}, which is for action spaces of type {policy.action_type}"
    wrong_type += f", on an action space of type {action_type}"
    assert policy.action_type == action_type, wrong_type
    assert algos[args.algo].is_learner, f"Ensure you call @learner on the {algos[args.algo]} class"

    if args.load:
        policy.load(f"./models/{file_name}")

    replay_size = args.ep_len if args.on_policy else args.buffer_size
    batch_size = args.ep_len if args.on_policy else args.batch_size
    replay_buffer = ReplayBuffer(state_dim, 
                                 action_dim,
                                 max_size=replay_size, 
                                 continuous=(action_type == ActionType.CONTINUOUS), 
                                 get_q=algos[args.algo].get_q,
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
        args.eval_freq
    )
    
    
    
    
    