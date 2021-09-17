# Tabular Q-Learning
Requirements: [[Value Functions]]

If you've read the requirements, you understand how [MDP]s can represent a very general class of tasks; namely, any task where an agent interacts with an environment. You also understand how the [[Value Functions | Q Function]] $Q(s, a)$ encodes the expected return when taking an action $a$ in a state $s$. We are now ready to formulate our first solution to MDPs with discrete state and action spaces!

## Strategy

Given an MDP $(S, A, \mathcal{T}, r)$, let's simply memorize the value of $Q(s, a)$ for each state $s$ and action $a$ we see. When our algorithm is running, it will take the action $a$ which maximizes our recoded $Q(s, a)$, or a random action if we have not recorded $Q(s, a)$ for any $a$.

Formally, we'll put them in a *table* indexed by $s$ and $a$ and containing estimates of our $Q$-values. Hence, this method is called **Tabular Q-Learning**. Let $\hat{Q}(s, a)$ be the estimate corresponding to state-action pair $(s, a)$. When the algorithm is queried at a state $s$, we will find the set $A(s)$ of actions $a$ for which $\hat{Q}(s, a)$ already exists:
$$A(s) = \{a \mid \hat Q(s, a) \text{ is recorded}\}$$
If $A(s)$ is empty, we take a random action. Otherwise, we take the action
$$a^*(s) = \mathop{\text{argmax}}_{a \in A(s)}\; \hat{Q}(s, a)$$

## Details

Assuming we can reliably calculate $Q$-values, this is a sensible way for our agent to interact in the environment to maximize expected reward. We do have one problem, however: we do not directly see the value of $Q(s, a)$ when taking an action $a$ in state $s$, so we do not know what value to record on our table. The [[Value Functions#Bellman Recurrence | Bellman Recurrence]] saves us here when combined with state-counting:
$$Q(s, a) = r(s, a) + \mathop{\mathbb{E}}_{s' \sim \mathcal{T}(s' \mid s, a)}\max_{a'\in A} Q(s', a')$$
This cannot be applied naively because we do not know the state transition dynamics $\mathcal{T}(s'\mid s, a)$, but it turns out we can estimate these dynamics. Let's say we are in a state $s$, we take an action $a$, and we land in state $s'$. Let $n(s, a)$ be a counter representing the number of times we have taken action $a$ in state $s$.  If $\hat{Q}(s, a)$ has no entry, let
$$\hat{Q}(s, a) = r(s, a) + \max_{a'\in A(s')} \hat{Q}(s', a')$$
$$n(s, a) = 1$$
Otherwise, update $Q(s, a)$ and $n(s, a)$ as
$$\hat{Q}(s, a) \leftarrow \frac{1}{n(s, a)}\left((n(s, a) - 1)\hat{Q}(s, a) + \max_{a'\in A(s')}\hat{Q}(s', a')\right)$$
$$n(s, a) \leftarrow n(s, a) + 1$$
Notice that our updates are designed to average over the second term in the [[Value Functions#Bellman Recurrence | Bellman Recurrence]], so the more we take an action $a$ in a state $s$, the more samples $s'$ we are averaging over, and the lower the variance of our estimate of $Q(s, a)$. 

## Analysis

**Theorem: Our algorithm converges if the MDP is finite and acyclic.**

Proof: First note that if we run TQL long enough, $A(s')$ will become the entire action space $A$, meaning we have visited $(s', a')$ for each $(s', a') \in S\times A$. This will come in handy later. 

We prove that TQL converges by induction on state-action pairs. We know TQL automatically converges on terminal states because we always visit these states (see the paragraph above), and convergence on these states takes only 1 visit. Suppose we also know that whenever TQL converges on all state-action pairs reachable from $(s, a)$, it also eventually converges on $(s, a)$. Then, because the MDP is finite and acylic, we can inductively establish that TQL converges on the entire MDP.

We now prove that TQL converges on some fixed $(s, a)$ using the assumption that it converges on all state-action pairs reachable from $(s, a)$.

Let $n = n(s, a)$, so we've visited this state-action pair $n$ times. Let $s'_1, s'_2, ..., s'_n$ be the states we landed on each time, and let $\hat{Q}_k(\cdot, \cdot)$ be the estimate in the table just after visiting $(s, a)$ and landing on $s'_k$. (By extension, $\hat{Q}_0(\cdot, \cdot)$ is the estimate before we ever visit $(s, a)$.) Then we can unroll our updates as follows:
$$\begin{align*}\hat{Q}(s, a) &= \hat{Q}_n(s, a)\\&=\frac{1}{n}\left((n - 1)\hat{Q}_{n-1}(s, a) + r(s, a) + \max_{a'\in A(s'_n)}\hat{Q}_{n-1}(s'_n, a')\right)\\&= \frac{1}{n}\left((n - 2)\hat{Q}_{n - 2}(s, a) + 2r(s, a) + \max_{a'\in A(s'_{n-1})}\hat{Q}_{n-2}(s'_{n-1}, a')+ \max_{a'\in A(s'_{n})}\hat{Q}_{n-1}(s'_{n}, a')\right)\\&= \frac{1}{n}\left[nr(s, a) + \sum_{k=1}^n\max_{a'\in A(s'_k)}\hat{Q}_{k-1}(s'_k, a')\right]\\&=r(s, a) + \frac{1}{n}\sum_{k=1}^n\max_{a'\in A(s'_k)}\hat{Q}_{k-1}(s'_k, a')\end{align*}$$
But recall we have $A(s'_k) = A$, and since TQL has converged on all state-action pairs reachable from $(s, a)$, we have $\hat{Q}_{k-1}(s'_k, a') = \hat{Q}(s'_k, a')$. Putting these together, we have shown
$$\hat{Q}(s, a) = r(s, a) + \frac{1}{n}\sum_{k=1}^n\max_{a'\in A}\hat{Q}(s'_k, a')$$
which is clearly an unbiased Monte-Carlo estimate of the Bellman expression when $s'_1, ..., s'_n \sim \mathcal{T}(s'\mid s, a)$.