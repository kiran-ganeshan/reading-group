# RL Paradigms
Requirements: [[PG]], [[DQN]], [[BC]]
## Introduction
The three basic RL algorithms we've seen illustrate an important division in how RL algorithms are trained.

Recall that when applying the [[PG|policy gradient]]
$$\nabla_{\theta}J(\theta) \propto \mathop{\mathbb{E}}_{s\sim\mu^{\pi}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi(s)}\Big[Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\Big]$$
we approximate the expectations using samples $(s_t, a_t)$ from trajectories gathered using the current policy $\pi$. Once the policy changes, we must gather new trajectories to train on. 

In contrast, recall that in [[DQN]], we only needed to approximate an expectation with respect to the transition function $\mathcal{T}(s'\mid s, a)$, which does not change during training. This allowed us to store old trajectories in a replay buffer and train on samples from the buffer. 

Unlike [[PG]] and [[DQN]], using [[BC|behavioral cloning]] to train a policy does not involve gathering trajectories. Rather, we train on a dataset of previous trajectories gathered by an expert.

## Definition
We refine the observations above into the following definitions of RL training paradigms:
- Algorithms like [[PG]] which can only train on trajectories from the current policy are called **on-policy**.
- Algorithms like [[DQN]] which can train on arbitrary trajectories are called **off-policy**.
- Algorithms like [[BC]] which are designed to train on a fixed dataset of previous trajectories are called **offline**.

Note that while BC requires that an expert produce the fixed dataset of trajectories, Offline RL algorithms in general may be designed to deal with suboptimal data.