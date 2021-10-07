import numpy as np
import torch
import gym
import argparse
import os
from torch_discounted_cumsum import discounted_cumsum_right
from torch import TensorType
from PG.pg import PG
 
 
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
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


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DQN")                      # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="CartPole-v1")                 # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)    # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)           # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)       # Max time steps to run environment
    parser.add_argument("--batch_size", default=256, type=int)          # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99)                     # Discount factor
    parser.add_argument("--tau", default=0.0005)                        # Target network update rate
    parser.add_argument("--eps", default=1e-2)
    parser.add_argument("--policy_freq", default=2, type=int)           # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")            # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                     # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_action = float(env.action_space.n)

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "discount": args.discount,
        "tau": args.tau,
        "eps": args.eps
    }

    policy = PG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    # Initialize training run
    wandb.init(project='gpt-3', entity='mlab-rl-benchmarking')
    config = wandb.config
    config.learning_rate = 0.01
    
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, args.env, args.seed)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    def get_new_traj():
        return (torch.zeros((0, state_dim)),
                torch.zeros((0,)),
                torch.zeros((0, state_dim)),
                torch.zeros((0,)),
                torch.zeros((0,)))
    data = get_new_traj()

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(torch.FloatTensor(state))

        # Perform action
        next_state, reward, done, _ = env.step(action) 
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        new_data = ()
        new_step = (state, action, next_state, reward, done)
        for d, n in zip(data, new_step):
            n = torch.unsqueeze(torch.tensor(n), axis=0)
            new_data = (*new_data, torch.cat([d, n], axis=0).type(torch.FloatTensor))
        data = new_data
        state = next_state
        episode_reward += reward

        if done: 
            
            # calculate q-values and insert them into data tuple
            qval = discounted_cumsum_right(torch.unsqueeze(data[3] * data[4], axis=1), kwargs['discount'])
            data = (*data[:4], qval, *data[4:])
            
            # train the policy 
            metrics = policy.train(*data)
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 
            data = get_new_traj()

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")