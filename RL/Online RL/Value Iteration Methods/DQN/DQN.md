# Deep Q-Network
Requirements: [[TQL]], [[Neural Network]]
## Resources
- The original paper: [Playing Atari with Deep Reinforcement Learning](Playing Atari with Deep Reinforcement Learning).
- Extensions of DQN: [SpinningUp Deep Q-Learning Reading List](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#a-deep-q-learning) (in particular, Double Q-Learning for preventing optimistic Q-values)
- Problems with DQN: [Rainbow](https://arxiv.org/abs/1710.02298)

## Introduction
You've seen how [[TQL]] can be used to solve MDPs. Recall that our $Q$ function is represented as an array/table of size $|S \times A|$. The larger our state and/or action space, the costlier this representation becomes. 

The core idea behing a Deep Q-network is to represent the $Q$ function with a function approximator, a deep neural network in this case. The associated algorithm, deep Q-learning, is similar to TQL in many ways except that we perform a gradient step to update the function.