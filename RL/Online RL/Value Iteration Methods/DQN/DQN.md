# Deep Q-Network
Requirements: [[TQL]], [[Neural Network]]

You've seen how [[TQL]] can be used to solve MDPs. Recall that our $Q$ function is represented as an array/table of size $|S \times A|$. The larger our state and/or action space, the costlier this representation becomes. 

The core idea behing a Deep Q-network is to represent the $Q$ function with a function approximator, a deep neural network in this case. The associated algorithm, deep Q-learning, is similar to TQL in many ways except that we perform a gradient step to update the function.

## Resources

* Read the original paper, [Playing Atari with Deep Reinforcement Learning](Playing Atari with Deep Reinforcement Learning). You have enough context to understand the paper!
* Take a look at Spinning Up's [Deep Q-learning reading list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#a-deep-q-learning) if you want to go further. In particular, double Q-learning is a common trick to prevent overly optimistic Q-values. It may also be instructional to look at [Rainbow](https://arxiv.org/abs/1710.02298) to get a sense for some of the pitfalls of vanilla Deep Q-networks.