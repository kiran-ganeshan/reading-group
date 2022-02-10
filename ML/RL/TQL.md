# Tabular Q-Learning
Requirements: [[Value Functions]]
## Resources
- [Sutton & Barto](http://incompleteideas.net/book/RLbook2020.pdf) 
	- Chapter 2 on Multi-Armed Bandits (equation 2.4 is important)
	- Chapters 5.2 - 5.3 on Monte Carlo $Q$ estimation
- [Medium Series on TQL for Tic Tac Toe](https://medium.com/@carsten.friedrich/part-3-tabular-q-learning-a-tic-tac-toe-player-that-gets-better-and-better-fa4da4b0892a)

## Introduction

If you've read the requirements, you understand how MDPs can represent a very general class of tasks; namely, any task where an agent interacts with an environment. You also understand how the [[Value Functions | Q Function]] $Q(s, a)$ encodes the expected return when taking an action $a$ in a state $s$. We are now ready to formulate our first solution to MDPs with discrete state and action spaces!

## Strategy

Given an MDP $(\sc S, \sc A, \mathcal{T}, r)$, let's simply memorize the value of $Q(s, a)$ for each state $s$ and action $a$ we see. When our algorithm is running, it will take the action $a$ which maximizes our recoded $Q(s, a)$, or a random action if we have not recorded $Q(s, a)$ for any $a$.

Formally, we'll put them in a *table* indexed by $s$ and $a$ and containing estimates of our $Q$-values. Hence, this method is called **Tabular Q-Learning**. Let $\hat{Q}(s, a)$ be the estimate corresponding to state-action pair $(s, a)$. When the algorithm is queried at a state $s$, we will find the set $A(s)$ of actions $a$ for which $\hat{Q}(s, a)$ already exists:
$$A(s) = \{a\in\sc A \mid \hat Q(s, a) \text{ is recorded}\}$$
If $A(s)$ is empty, we take a random action. Otherwise, we take the action
$$a^*(s) = \mathop{\text{argmax}}_{a \in A(s)}\; \hat{Q}(s, a)$$

In order to update the $\hat Q$ table as we see new transitions, suppose we experience a transition $(s, a, s')$ $$n(s, a) \leftarrow n(s, a) + 1$$
$$\hat{Q}(s, a) \leftarrow \frac{1}{n(s, a)}\bigg((n(s, a) - 1)\hat{Q}(s, a) + r(s, a) + \gamma\max_{a'\in A(s')}\hat{Q}(s', a')\bigg)$$
These updates may be described as ***continual averaging***: they are designed such that $\hat{Q}(s, a)$ is the average (over $s'$) of the target value $r(s, a) + \gamma\max_{a'\in A(s')} \hat{Q}(s', a')$. Storing the count $n(s, a)$ allows us to simply store but we don't have to keep track of every target $Q$ value for each $s'$ in order to calculate this average (we can do this by storing the average itself as well as the number of samples, hence why we introduce $n(s, a)$). The more we take an action $a$ in a state $s$, the more samples $s'$ we are averaging over, and the lower the variance of our estimate of $Q(s, a)$. 

## Use Cases
TQL can be used to solve MDPs with 
- Discrete and Finite State Spaces
- Discrete and Finite Action Spaces

## Algorithm

Let $\hat Q$ be a table with entries $\hat Q(s, a)$ for each $s\in S$ and $a\in A$. Let $n(s, a)$ be a counter representing the number of times we have taken action $a$ in state $s$. 

Suppose we are in a state $s$, we take an action $a$, and we land in state $s'$.  If $\hat{Q}(s, a)$ has no entry, let
$$\hat{Q}(s, a) = r(s, a) + \gamma\max_{a'\in A(s')} \hat{Q}(s', a')$$
$$n(s, a) = 1$$
Otherwise, update $Q(s, a)$ and $n(s, a)$ as

$$n(s, a) \leftarrow n(s, a) + 1$$
$$\hat{Q}(s, a) \leftarrow \frac{1}{n(s, a)}\bigg((n(s, a) - 1)\hat{Q}(s, a) + r(s, a) + \gamma\max_{a'\in A(s')}\hat{Q}(s', a')\bigg)$$
These updates may be described as ***continual averaging***: they are designed such that $\hat{Q}(s, a)$ is the average (over $s'$) of the target value $r(s, a) + \gamma\max_{a'\in A(s')} \hat{Q}(s', a')$. Storing the count $n(s, a)$ allows us to simply store the running average rather than keeping track of every target $Q$ value we see to calculate this average. The more we take an action $a$ in a state $s$, the more samples $s'$ we are averaging over, and the lower the variance of our estimate of $Q(s, a)$. 

## Analysis

**Theorem: Our algorithm converges if the MDP is finite and acyclic.**

Proof: First note that if we run TQL long enough, $A(s')$ will become the entire action space $\sc A$, meaning we have visited $(s', a')$ for each $(s', a') \in \sc S\times \sc A$. This will come in handy later. 

We prove that TQL converges by induction on state-action pairs. We know TQL automatically converges on terminal states because we always visit these states (see the paragraph above), and convergence on these states takes only 1 visit. Suppose we also know that whenever TQL converges on all state-action pairs reachable from $(s, a)$, it also eventually converges on $(s, a)$. Then, because the MDP is finite and acylic, we can inductively establish that TQL converges on the entire MDP.

We now prove that TQL converges on some fixed $(s, a)$ using the assumption that it converges on all state-action pairs reachable from $(s, a)$.

Let $n = n(s, a)$, so we've visited this state-action pair $n$ times. Let $s'_1, s'_2, ..., s'_n$ be the states we landed on each time, and let $\hat{Q}_k(\cdot, \cdot)$ be the estimate in the table just after visiting $(s, a)$ and landing on $s'_k$. (By extension, $\hat{Q}_0(\cdot, \cdot)$ is the estimate before we ever visit $(s, a)$.) Then we can unroll our updates as follows:
$$\begin{align*}\hat{Q}(s, a) &= \hat{Q}_n(s, a)\\&=\frac{1}{n}\bigg((n - 1)\hat{Q}_{n-1}(s, a) + r(s, a) + \gamma\max_{a'\in A(s'_n)}\hat{Q}_{n-1}(s'_n, a')\bigg)\\&=\hat{Q}_n(s, a)\\&=\frac{1}{n}\bigg[(n - 1)\frac{1}{n-1}\bigg((n - 2)\hat Q_{n-2}(s, a) + r(s, a) + \gamma\max_{a'\in A(s'_{n-1})}\hat{Q}_{n-2}(s'_{n-1}, a')\bigg) + r(s, a) + \gamma\max_{a'\in A(s'_n)}\hat{Q}_{n-1}(s'_n, a')\bigg]\\&= \frac{1}{n}\bigg((n - 2)\hat{Q}_{n - 2}(s, a) + 2r(s, a) + \gamma\max_{a'\in A(s'_{n-1})}\hat{Q}_{n-2}(s'_{n-1}, a')+ \gamma\max_{a'\in A(s'_{n})}\hat{Q}_{n-1}(s'_{n}, a')\bigg)\\&\vdots\\&= \frac{1}{n}\Bigg[nr(s, a) + \sum_{k=1}^n\gamma\max_{a'\in A(s'_k)}\hat{Q}_{k-1}(s'_k, a')\Bigg]\\&=r(s, a) + \gamma\Bigg[\frac{1}{n}\sum_{k=1}^n\max_{a'\in A(s'_k)}\hat{Q}_{k-1}(s'_k, a)\Bigg]\end{align*}$$
But recall we have $A(s'_k) = \sc A$, and since TQL has converged on all state-action pairs reachable from $(s, a)$, we have $\hat{Q}_{k-1}(s'_k, a') = \hat{Q}(s'_k, a')$. Putting these together, we have shown
$$\hat{Q}(s, a) = r(s, a) + \frac{1}{n}\sum_{k=1}^n\max_{a'\in \sc A}\hat{Q}(s'_k, a')$$
which is clearly an unbiased Monte-Carlo estimate of the Bellman expression when $s'_1, ..., s'_n \sim \mathcal{T}(s'\mid s, a)$:
$$\hat{Q}(s, a) = r(s, a) + \mathop{\mathbb{E}}_{s'\sim \mathcal{T}(s'\mid s, a)}\bigg[\max_{a'\in A}\hat{Q}(s', a')\bigg]$$
