# Value Functions
Requirements: [[MDP]]
## Resources
- [Sutton & Barto 3.5 - 3.8](http://incompleteideas.net/book/RLbook2020.pdf)
- [ML@B Blog on MDPs](https://ml.berkeley.edu/blog/posts/mdps/)
- [Towards Data Science Series on MDPs](https://towardsdatascience.com/introduction-to-reinforcement-learning-markov-decision-process-44c533ebf8da)

## Introduction
Note that in the definition of the MDP, we would like to maximize expected reward *across entire trajectories*. Suppose we run two agents for a million time steps on a single MDP. One receives reward 0.5 on every state transition, and the other receives no reward until the last transition, on which it recieves 600,000 reward. Even though it takes the second agent $>10^5$ steps to get any reward, while the first gets reward immediately, the second agent has higher expected reward, so it is the preferred solution.

Clearly, we cannot solve MDPs by teaching agents to maximize immediate reward, so it helps to have an abstract way of discussing the expected reward-to-go. For this purpose, we define the following functions:
1. We define $Q^{\pi}(s, a)$ to be the expected reward-to-go when taking action $a$ in state $s$ under policy $\pi$.
2. We define $Q^*(s, a)=\max_{\pi}Q^{\pi}(s, a)$ to be the expected reward-to-go under the optimal policy.
3. We define $V^{\pi}(s)$ to be the expected reward-to-go when in state $s$ acting under policy $\pi$. 
4. We define $V^*(s) = \max_{\pi}V^{\pi}(s)$ to be the expected reward-to-go under the optimal policy.

## Definition
For an MDP with finite horizon $T\in\mathbb{N}$:
$$\begin{align*}Q^*(s, a) &= \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau\mid s, a)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right]\\Q^{\pi}(s, a) &= \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau \mid s, a)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right]\\V^*(s) &= \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau \mid s)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right] \\V^{\pi}(s) &= \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau \mid s)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right]\end{align*}$$
(Recall the distributions over [[MDP#Describing Solutions | trajectories]] $\mathcal{T}^{\pi}$ and $\mathcal{T}^*$.) These expressions represent the expected total reward under the optimal policy and under policy $\pi$, respectively. To see why these may be useful, note that since our objective is simply the expected total reward, the optimal policy is given by
$$\pi^*(s) = \mathop{\text{argmax}}_{a\in A}\; Q^*(s, a)$$
And thus learning the $Q$-values under the optimal policy can be helpful in determining the optimal policy.

## Infinite-Horizon Case
When we are in an [[MDP#Infinite Horizon MDPs| Infinite Horizon MDP]], we've seen that we add a discount factor:
$$\begin{align*}Q^*(s, a) &= \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(s, a)}\left[\sum_{t=0}^{\infty}\gamma^tr(s_t, a_t)\right]\\
Q^{\pi}(s, a) &= \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(s, a)}\left[\sum_{t=0}^{\infty}\gamma^tr(s_t, a_t)\right]\\
V^*(s) &= \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(s)}\left[\sum_{t=0}^{\infty}\gamma^tr(s_t, a_t)\right]\\
V^{\pi}(s) &=  \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(s)}\left[\sum_{t=0}^{\infty}\gamma^tr(s_t, a_t)\right]\end{align*}$$
Since the optimal policy in an Infinite Horizon MDP is the one which maximizes expected discounted total reward, we again have
$$\pi^*(s) = \mathop{\text{argmax}}_{a\in A}\; Q^*(s, a)$$
despite that fact that $\pi^*$ and $Q^*$ are defined slightly differently than in the finite horizon case. This is similar to the reason we can still turn expectations over the optimal policy into maximums when working with discounted reward: recall [[MDP#^0c1458 | this paragraph]].


## Expectation Recurrence
Now recall the [[MDP#Decomposing Expectations | Expectation Decompositions]] for expectations over trajectory distributions:
$$\begin{align*}\mathop{\mathbb{E}}_{\tau \sim \mathcal{T}^{\pi}(\tau \mid s)} &= \mathop{\mathbb{E}}_{a_0\sim \pi(s)}\;\boxed{\mathop{\mathbb{E}}_{s_1\sim \mathcal{T}(s_1\mid s, a_0)}\;\mathop{\mathbb{E}}_{a_1\sim \pi(s_1)}\;\mathop{\mathbb{E}}_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)} \dots}\\
\mathop{\mathbb{E}}_{\tau \sim \mathcal{T}^{\pi}(\tau \mid s, a)} &=\boxed{ \mathop{\mathbb{E}}_{s_1\sim \mathcal{T}(s_1\mid s, a)}\;\mathop{\mathbb{E}}_{a_1\sim \pi(s_1)}\;\mathop{\mathbb{E}}_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)}\;\mathop{\mathbb{E}}_{a_2\sim \pi(s_2)} \dots }\\\mathop{\mathbb{E}}_{\tau \sim \mathcal{T}^*(\tau \mid s)} &= \mathop{\mathbb{E}}_{a_0\sim \pi^*(s)}\;\boxed{\mathop{\mathbb{E}}_{s_1\sim \mathcal{T}(s_1\mid s, a_0)}\;\mathop{\mathbb{E}}_{a_1\sim \pi^*(s_1)}\;\mathop{\mathbb{E}}_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)} \dots}\\
\mathop{\mathbb{E}}_{\tau \sim \mathcal{T}^*(\tau \mid s, a)} &=\boxed{ \mathop{\mathbb{E}}_{s_1\sim \mathcal{T}(s_1\mid s, a)}\;\mathop{\mathbb{E}}_{a_1\sim \pi^*(s_1)}\;\mathop{\mathbb{E}}_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)}\;\mathop{\mathbb{E}}_{a_2\sim \pi^*(s_2)} \dots}\end{align*}$$
Notice the similarity between the parts of the expressions we boxed. They allow us to write the following:
$$\begin{align*}\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau\mid s)} &= \mathop{\mathbb{E}}_{a\sim\pi(s)}\; \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau\mid s, a)}\\\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau\mid s)} &= \mathop{\mathbb{E}}_{a\sim\pi^*(s)}\; \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau\mid s, a)}\end{align*}$$
(Note: In a Finite Horizon MDP, the boxed expressions start with the same expectations, but the expressions corresponding to $\mathcal{T}^{\pi}(\tau\mid s, a)$ and $\mathcal{T}^*(\tau\mid s, a)$ may not have the same number of expectations as those corresponding to $\mathcal{T}^{\pi}(\tau\mid s)$ and $\mathcal{T}^*(\tau\mid s)$. For convenience, we will assume we are dealing with an Infinite Horizon MDP. We will see that it is trivial to extend our results to Finite Horizon MDPs.) We can also see a similarity between the following boxed expressions:
$$\begin{align*}\mathop{\mathbb{E}}_{\tau \sim \mathcal{T}^{\pi}(\tau \mid s)} &=\boxed{ \mathop{\mathbb{E}}_{a_0\sim \pi(s)}\;\mathop{\mathbb{E}}_{s_1\sim \mathcal{T}(s_1\mid s, a_0)}\;\mathop{\mathbb{E}}_{a_1\sim \pi(s_1)}\;\mathop{\mathbb{E}}_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)} \dots}\\
\mathop{\mathbb{E}}_{\tau \sim \mathcal{T}^{\pi}(\tau \mid s, a)} &= \mathop{\mathbb{E}}_{s_1\sim \mathcal{T}(s_1\mid s, a)}\;\boxed{\mathop{\mathbb{E}}_{a_1\sim \pi(s_1)}\;\mathop{\mathbb{E}}_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)}\;\mathop{\mathbb{E}}_{a_2\sim \pi(s_2)} \dots }\\\mathop{\mathbb{E}}_{\tau \sim \mathcal{T}^*(\tau \mid s)} &= \boxed{\mathop{\mathbb{E}}_{a_0\sim \pi^*(s)}\;\mathop{\mathbb{E}}_{s_1\sim \mathcal{T}(s_1\mid s, a_0)}\;\mathop{\mathbb{E}}_{a_1\sim \pi^*(s_1)}\;\mathop{\mathbb{E}}_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)} \dots}\\
\mathop{\mathbb{E}}_{\tau \sim \mathcal{T}^*(\tau \mid s, a)} &= \mathop{\mathbb{E}}_{s_1\sim \mathcal{T}(s_1\mid s, a)}\;\boxed{\mathop{\mathbb{E}}_{a_1\sim \pi^*(s_1)}\;\mathop{\mathbb{E}}_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)}\;\mathop{\mathbb{E}}_{a_2\sim \pi^*(s_2)} \dots}\end{align*}$$
which give rise to the following relations:
$$\begin{align*}\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau\mid s, a)} &= \mathop{\mathbb{E}}_{s'\sim\mathcal{T}^{\pi}(s, a)}\; \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau\mid s')}\\\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau\mid s, a)} &= \mathop{\mathbb{E}}_{s'\sim\mathcal{T}^*(s, a)}\; \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau\mid s')}\end{align*}$$
These expectation recurrence relations will help us derive recurrence relations for $Q^{\pi}$, $Q^*$, $V^{\pi}$, and $V^*$, since they are defined using expectations over trajectories. But we are taking the expectation of total (discounted) reward, which is the objective that our optimal policy $\pi^*$ maximizes. This allows us to substitute $\underset{a_i\in A}{\max}$ for $\underset{a_i\sim \pi^*(s_i)}{\mathbb{E}}$ in our decomposition, altering one of our recurrences:
$$\begin{align*}\mathop{\mathbb{E}}_{\tau \sim \mathcal{T}^*(\tau \mid s)} &= \max_{a_0\in A}\;\boxed{\mathop{\mathbb{E}}_{s_1\sim \mathcal{T}(s_1\mid s, a_0)}\;\max_{a_1\in A}\;\mathop{\mathbb{E}}_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)} \dots}\\
\mathop{\mathbb{E}}_{\tau \sim \mathcal{T}^*(\tau \mid s, a)} &= \boxed{\mathop{\mathbb{E}}_{s_1\sim \mathcal{T}(s_1\mid s, a)}\;\max_{a_1\in A}\;\mathop{\mathbb{E}}_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)}\;\max_{a_2\in A} \dots}\end{align*}$$
$$\Longrightarrow \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau\mid s)}=\max_{a\in A}\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(s, a)}$$
In conclusion, in Infinite-Horizon MDPs we can write
$$\begin{align*}\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau\mid s, a)} &= \mathop{\mathbb{E}}_{s'\sim\mathcal{T}^{\pi}(s, a)}\; \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau\mid s')}\\\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau\mid s, a)} &= \mathop{\mathbb{E}}_{s'\sim\mathcal{T}^*(s, a)}\; \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau\mid s')}\\\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau\mid s)} &= \mathop{\mathbb{E}}_{a\sim\pi(s)}\; \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau\mid s, a)}\\\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau\mid s)}&=\max_{a\in A}\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(s, a)}\end{align*}$$
We will use these to derive recurrence relations which relate $V$ to $Q$ and $Q$ to $V$. Putting these partial recurrence relations together, we can derive full recurrence relations which relate $V$ to $V$ and $Q$ to $Q$.

## Bellman Recurrence
Using the expectation recurrence relations we derived above (the last of which is only valid when we are applying it to the RL objective, as we are in $Q$ and $V$), we can derive the following partial recurrence relations:

|  | Partial Recurrence Relation |
| --- | --- | 
|$Q^*$| $$\begin{align*}Q^*(s, a) &= \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau \mid s, a)}\left[\sum_{t=0}^{\infty}\gamma^tr(s_t, a_t)\right] &\\&= \mathop{\mathbb{E}}_{s'\sim \mathcal{T(s' \mid s,a)}}\;\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau \mid s')}\left[\gamma^0r(s, a) + \sum_{t=0}^{\infty}\gamma^{t+1}r(s_t, a_t)\right]& \\&= \mathop{\mathbb{E}}_{s'\sim \mathcal{T(s' \mid s,a)}}\left\{r(s, a) + \gamma\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau \mid s')}\left[\sum_{t=0}^{\infty}\gamma^tr(s_t, a_t)\right]\right\}&\\&= r(s, a) + \gamma\mathop{\mathbb{E}}_{s'\sim \mathcal{T(s' \mid s,a)}} V^*(s')&\end{align*}$$ |
| $Q^{\pi}$ | $$\begin{align*}Q^{\pi}(s, a) &= \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau \mid s)}\left[\sum_{t=0}^{\infty}r(s_t, a_t)\right]\\&=\mathop{\mathbb{E}}_{s'\sim \mathcal{T}(s' \mid s,a)}\; \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau \mid s')}\left[\gamma^0r(s, a) + \sum_{t=0}^{T}\gamma^{t+1}r(s_t, a_t)\right]\\&= \mathop{\mathbb{E}}_{s'\sim \mathcal{T}(s' \mid s,a)}\left\{r(s, a) + \gamma\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau \mid s, a)}\left[\sum_{t=0}^{\infty}\gamma^tr(s_t, a_t)\right]\right\}\\&=r(s, a) + \gamma\mathop{\mathbb{E}}_{s'\sim \mathcal{T}(s' \mid s,a)}\;V^{\pi}(s')\end{align*} $$ |
| $V^*$ | $$\begin{align*}V^*(s) &= \left(\frac{1}{1-\gamma}\right)\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau \mid s)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right] & \\&=\left(\frac{1}{1-\gamma}\right) \max_{a\in A}\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau \mid s, a)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right] &\\&= \max_{a\in A}\left(\frac{1}{1-\gamma}\right)\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^*(\tau \mid s, a)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right]&\\&= \max_{a\in A} Q^*(s, a)\end{align*}$$ |
| $V^{\pi}$ | $$\begin{align*}V^{\pi}(s) &= \left(\frac{1}{1-\gamma}\right)\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau \mid s)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right]\\&=\left(\frac{1}{1-\gamma}\right) \mathop{\mathbb{E}}_{a\sim \pi(a\mid s)}\;\mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau \mid s, a)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right] \\&=\mathop{\mathbb{E}}_{a\sim \pi(a\mid s)}\left(\frac{1}{1-\gamma}\right) \mathop{\mathbb{E}}_{\tau\sim\mathcal{T}^{\pi}(\tau \mid s, a)}\left[\sum_{t=0}^{T}r(s_t, a_t)\right]\\&=\mathop{\mathbb{E}}_{a\sim \pi(a\mid s)}\;Q^{\pi}(s, a)\end{align*}$$ |

^4e628e

We can plug these partial recurrence relations into one another to obtain the following full recurrence relations for Infinite Horizon MDPs:

|  | Full Recurrence Relation |
| --- | --- | 
|$Q^*$| $$\begin{align*}Q^*(s, a) &= r(s, a) + \gamma\mathop{\mathbb{E}}_{s'\sim \mathcal{T(s' \mid s,a)}} V^*(s')\\&=r(s, a) + \gamma\mathop{\mathbb{E}}_{s'\sim \mathcal{T(s' \mid s,a)}}\max_{a\in A}Q^*(s, a)\end{align*}$$ |
| $Q^{\pi}$ | $$\begin{align*}Q^{\pi}(s, a) &=r(s, a) + \gamma\mathop{\mathbb{E}}_{s'\sim \mathcal{T}(s' \mid s,a)}\;V^{\pi}(s')\\&=r(s, a) + \gamma\mathop{\mathbb{E}}_{s'\sim \mathcal{T}(s' \mid s,a)}\;\mathop{\mathbb{E}}_{a\sim \pi(a\mid s)}\;Q^{\pi}(s, a)\end{align*} $$ |
| $V^*$ | $$\begin{align*}V^*(s) &=\max_{a\in A}Q^*(s, a)\\&= \max_{a\in A} \left\{r(s, a) + \gamma\mathop{\mathbb{E}}_{s'\sim \mathcal{T(s' \mid s,a)}} V^*(s')\right\}\end{align*}$$ |
| $V^{\pi}$ | $$\begin{align*}V^{\pi}(s) &=\mathop{\mathbb{E}}_{a\sim \pi(a\mid s)}\;Q^{\pi}(s, a)\\&=\mathop{\mathbb{E}}_{a\sim \pi(a\mid s)}\left\{r(s, a) + \gamma\mathop{\mathbb{E}}_{s'\sim \mathcal{T}(s' \mid s,a)}\;V^{\pi}(s')\right\}\end{align*}$$ |



