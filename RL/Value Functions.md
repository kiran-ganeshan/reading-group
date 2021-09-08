# Value Functions
Requirements: [[MDP]]

## Introduction
Note that in the definition of the MDP, we would like to maximize expected reward *across entire trajectories*. Suppose we run two agents for a million time steps on a single MDP. One receives reward 0.5 on every state transition, and the other receives no reward until the last transition, on which it recieves 600,000 reward. Even though it takes the second agent $>10^5$ steps to get any reward, while the first gets reward immediately, the second agent has higher expected reward, so it is the preferred solution.

Clearly, we cannot solve MDPs by teaching agents to maximize immediate reward, so it helps to have an abstract way of discussing the expected reward-to-go. For this purpose, we define the following functions:
1. We define $Q^{\pi}(s, a)$ to be the expected total reward when taking action $a$ in state $s$ under policy $\pi$.
2. We define $Q^*(s, a)$ to be the expected total reward when taking action $a$ in state $s$ under the optimal policy $\pi^*$. 
3. We define $V^{\pi}(s)$ to be the expected total reward when in state $s$ acting under policy $\pi$.when taking an action $a$ in state $s$. We define  to be this expected reward-to-go when acting under a policy $\pi$, and we define $Q^*(s, a) = \max_{\pi}Q^{\pi}(s, a)$ to be the expected reward-to-go under the optimal policy.

## Definition
For an MDP where trajectories are at most $T$ time steps in length,
$$Q^*(s, a) = \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}(\tau\mid s, a)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right]$$
$$Q^{\pi}(s, a) = \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau \mid s, a)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right]$$
$$V^*(s) = \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau \mid s)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right] = \max_{a\in A}\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau \mid s, a)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right] = \max_{a\in A} Q^*(s, a)$$
$$V^{\pi}(s) = \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau \mid s)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right] = \mathop{\mathbb{E}}_{a\sim \pi(a\mid s)}\;\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau \mid s, a)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right] = \mathop{\mathbb{E}}_{a\sim \pi(a\mid s)}\;Q^{\pi}(s, a)$$
where clearly $\tau = ((s_1, a_1), \dots, (s_T, a_T))$. These expressions simply represent the expected total reward under the optimal policy and under policy $\pi$, respectively. Since our objective is simply the expected total reward, the optimal policy is given by
$$\pi^*(s) = \mathop{\text{argmax}}_{a\in A}\; Q^*(s, a)$$
And thus learning the $Q$-values under the optimal policy can be helpful in determining the optimal policy.

## Infinite-Horizon Case
The $Q$-values we have defined thus far are unproblematic in finite-horizon MDPs, where total reward is necessarily bounded. But what if we care about reward infinitely far into the future? Then it is not clear the $Q$-value expressions above converge. In this case, we add a discount factor $\gamma \in (0, 1)$ to future rewards:
$$Q^*(s, a) = \left(\frac{1}{1-\gamma}\right)\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}}\left[\sum_{t=0}^{T}\gamma^tr(s_t, a_t)\right]$$
$$Q^{\pi}(s, a) = \left(\frac{1}{1-\gamma}\right) \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}}\left[\sum_{t=0}^{T}\gamma^tr(s_t, a_t)\right]$$
Then, if we have $r(s, a) \leq R$ for some finite $R$ and all $(s, a)\in S\times A$, we have $Q^*(s, a)\leq R$ and $Q^{\pi}(s, a)\leq R$, so both expressions are bounded.

This discount factor turns out to be quite important for the stability of methods relying on value iteration (i.e., learning $Q$ or $V$ values). However, it also poses a problem for situations with sparse rewards. Even for values of $\gamma$ very close to 1, when $t$ grows large enough, $\gamma^t$ diminishes, meaning rewards that are far in the future are heavily discounted. The difficulty that naive value iteration experiences in the sparse-reward setting has prompted much research and development in exploration techniques for RL algorithms.

## Bellman Recurrence


