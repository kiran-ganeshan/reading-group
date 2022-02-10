 # Markov Decision Processes
Requirements: None
## Resources
- [Sutton & Barto 3.1 - 3.4](http://incompleteideas.net/book/RLbook2020.pdf)
- [ML@B Blog](https://ml.berkeley.edu/blog/posts/mdps/)
- [Towards Data Science](https://towardsdatascience.com/introduction-to-reinforcement-learning-markov-decision-process-44c533ebf8da)

## Introduction
One goal of the field of Artificial Intelligence is to automate certain tasks which are currently completed by humans. To approach this problem, it is helpful to be able to generalize these distinct tasks into a single framework. 

To understand how to choose the proper framework, let's think about the limitations of the machines we will attempt to use to solve tasks in this framework. 

First of all, unlike humans, whose internal neural dynamics respond continuously to the environment, computers can only respond at discrete time steps. 

Secondly, it is unclear how to communicate the intended task to our computers; they don't understand natural language by default. Even if we try to teach them how to speak our language, natural language is complex and misinterpretable, and we don't want our computers to act like the Genie from Aladdin when a user isn't specific enough. We will thus need a simpler signal to tell the computer when it is succeeding and failing at the task.

With these drawbacks in mind, we propose the **Markov Decision Process** (MDP). In this model, we have an **agent** interacting with an **environment** over discrete time steps, attempting to maximize its expected **reward**. At any discrete time step, three things happen in immediate succession:
1. The agent is in a **state**
2. The agent takes an **action**, moving it to a new state
3. The agent receives a numerical **reward** corresponding to the state and action in parts (1) and (2)

As this cycle repeats, the agent traces out a **trajectory** $\tau = ((s_0, a_0), \dots, (s_t, a_t))$, or list of states and actions, in the MDP. We use discrete time steps since our computers respond discretely to their surroundings, and we use a simple numerical reward as the representation of the task in order to give the computer a simple signal that both the computer and onlooking humans can understand.

## Definition

More concretely, An MDP is a tuple $(\sc S, \sc A, \mathcal{T}, r)$, meaning it consists of 
1. A **State Space** $\sc S$, the set of all of the states in which the agent could possibly be
2. An **Action Space** $\sc A$, the set of all the actions the agent could possibly take (sometimes a function of state)
3. A **Transition Operator** $\mathcal{T} : \sc S\times \sc A \to \mathcal{D}(\sc S)$, which defines the distribution $\mathcal{T}(s' \mid s, a)$ over possible next states $s'$ given the current state $s$ and the action $a$ just taken
4. A **Reward Function** $r: \sc S\times \sc A\to \mathbb{R}$, which determines the reward $r(s, a)$ received by the agent when it takes action $a$ in state $s$

Note that the reward function only depends on the current state-action pair $(s_t, a_t)$, and not the previous states or actions in our current trajectory. This is known as the **Markov Property** and is what distinguishes a regular old Decision Process from an MDP.

### Describing Solutions
To *solve* an MDP, we would like to know which actions to take in order to maximize expected total reward over many timesteps. Formally, a possible solution to an MDP is represented by a **policy** $\pi: \sc S\to \mathcal{D}(\sc A)$. The policy tells us the distribution over actions taken by the agent: at each time step, when the agent with policy $\pi$ begins in a state $s$, it will sample an action $a\sim\pi(s)$ from the policy and take this action. 

A path taken across an MDP is formalized as a **trajectory**, or a sequence of state-action pairs
$$\tau = ((s_0, a_0), (s_1, a_1), \dots (s_T, a_T))$$
The trajectory shown has finite **horizon** $T$, meaning the number of steps we take in the environment. A policy $\pi$ induces a distribution $\mathcal{T}^{\pi}$ over trajectories:
$$\mathcal{T}^{\pi}(\tau \mid s_0) = \prod_{k=0}^{T}\pi(a_k\mid s_k) \mathcal{T}(s_{k+1}\mid s_k, a_k)$$
$$\mathcal{T}^{\pi}(\tau \mid s_0, a_0) = \prod_{k=0}^{T} \mathcal{T}(s_{k+1}\mid s_k, a_k)\pi(a_{k+1}\mid s_{k+1})$$
These equations effectively define sampling processes: the first says, starting with $s_0$, first sample $a_0\sim \pi(a_0\mid s_0)$, then sample $s_1 \sim \mathcal{T}(s_1\mid s_0, a_0)$, then sample $a_1 \sim \pi(a_1\mid s_1)$, and so on. This sampling process gives rise to the distribution $\mathcal{T}^{\pi}(\tau\mid s_0)$. Similarly, the sampling process beginning with $s_1\sim\mathcal{T}(s_1\mid s_0, a_0)$ gives rise to the distribution $\mathcal{T}^{\pi}(\tau\mid s_0, a_0)$.

Using these distributions, we can write the maximization objective corresponding to this MDP with horizon $T$ as
$$\max_{\pi}\E_{\tau\sim \mathcal{T}^{\pi}(\tau\mid s)}\sum_{t=0}^T r(s_t, a_t)$$

Note there is some optimal policy $\pi^*$ which maximizes this objective. We let $\mathcal{T}^*$ refer to the distribution over trajectories under the optimal policy, so $\mathcal{T}^* = \mathcal{T}^{\pi^*}$. Then we have
$$\mathcal{T}^*(\tau \mid s_0) = \prod_{k=0}^{\infty}\pi^*(a_k\mid s_k) \mathcal{T}(s_{k+1}\mid s_k, a_k)$$
$$\mathcal{T}^*(\tau \mid s_0, a_0) = \prod_{k=0}^{\infty} \mathcal{T}(s_{k+1}\mid s_k, a_k)\pi^*(a_{k+1}\mid s_{k+1})$$

### Decomposing Expectations
Notice that our maximization objective involves an expectation over the distribution of trajectories $\mathcal{T}^{\pi}(\tau\mid s)$. Recall a trajectory $\tau$ is a list of states and actions:
$$\tau = ((s, a_0), (s_1, a_1), \dots (s_T, a_T))$$
where $s$ is given. Hence, taking an expectation over a distribution over $\tau$ is identical to taking an expectation over $a_0, s_1, a_1, \dots$ and so on:
$$\E_{\tau\sim\mathcal{T}(\tau\mid s)} =\underset{\substack{a_0 \sim \pi(a_0\mid s)\\s_1\sim \mathcal{T}(s_1\mid s, a_0)\\a_1 \sim \pi(a_1\mid s_1)\\\vdots}}{\mathbb{E}} = \E_{a_0 \sim \pi(a_0\mid s)}\;\E_{s_1\sim \mathcal{T}(s_1\mid s, a_0)}\;\E_{a_1\sim \pi(a_1\mid s_1)}\dots$$
We are able to decompose the expectation over trajectories into this series of expectations over states and actions because 
- the distribution of $a_0$ only depends on the value of $s$,
- the distribution of $s_1$ depends only on $s$ and $a_0$, 
- the distribution of $a_1$ depends only on $s_1$, and so on.

More generally, we can decompose an expectation over $\mathcal{T}^{\pi}$ into **alternating** expectations over $\pi$ and $\mathcal{T}$:
$$\begin{align*}\E_{\tau \sim \mathcal{T}^{\pi}(\tau \mid s)} &= \E_{a_0\sim \pi(s)}\;\E_{s_1\sim \mathcal{T}(s_1\mid s, a_0)}\;\E_{a_1\sim \pi(s_1)}\;\E_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)} \dots\\
\E_{\tau \sim \mathcal{T}^{\pi}(\tau \mid s, a)} &= \E_{s_1\sim \mathcal{T}(s_1\mid s, a)}\;\E_{a_1\sim \pi(s_1)}\;\E_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)}\;\E_{a_2\sim \pi(s_2)} \dots\end{align*}$$
Similarly, we can break an expectation over $\mathcal{T}^*$ into **alternating** expectations over $\pi^*$ and $\mathcal{T}$:
$$\begin{align*}\E_{\tau \sim \mathcal{T}^*(\tau \mid s)} &= \E_{a_0\sim \pi^*(s)}\;\E_{s_1\sim \mathcal{T}(s_1\mid s, a_0)}\;\E_{a_1\sim \pi^*(s_1)}\;\E_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)} \dots\\
\E_{\tau \sim \mathcal{T}^*(\tau \mid s, a)} &= \E_{s_1\sim \mathcal{T}(s_1\mid s, a)}\;\E_{a_1\sim \pi^*(s_1)}\;\E_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)}\;\E_{a_2\sim \pi^*(s_2)} \dots\end{align*}$$
In most situations we don't know $\pi^*$, so this decomposition is useless. However, in the original maximization objective, we are taking the expectation of the reward-to-go, and $\pi^*$ is the policy which maximizes the reward-to-go. Therefore we can write
$$\begin{align*}\E_{\tau \sim \mathcal{T}^*(\tau \mid s)}\Bigg[\sum_{t=0}^Tr(s_t, a_t)\Bigg] &= \max_{a_0\in \sc A}\;\E_{s_1\sim \mathcal{T}(s_1\mid s, a_0)}\;\max_{a_1\in \sc A}\;\E_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)} \dots\Bigg[\sum_{t=0}^Tr(s_t, a_t)\Bigg]\\
\E_{\tau \sim \mathcal{T}^*(\tau \mid s, a)}\Bigg[\sum_{t=0}^Tr(s_t, a_t)\Bigg] &= \E_{s_1\sim \mathcal{T}(s_1\mid s, a)}\;\max_{a_1\in \sc A}\;\E_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)}\;\max_{a_2\in \sc A} \dots\Bigg[\sum_{t=0}^Tr(s_t, a_t)\Bigg]\end{align*}$$

### Sampling from Trajectories

Suppose we wanted to sample individual states from our trajectories. When we build learning algorithms to solve MDPs, this is necessary since our learning algorithms will train on previously seen states. Given that our trajectories start at a state $s_0$, what di

### Infinite Horizon MDPs

Sometimes, we would like find a policy which maximizes reward-to-go infintely far in the future: we would like to set the horizon $T = \infty$. Note that then the expected total rewards over our infintely long trajectories can grow unboundedly, so there may be no policy $\pi$ which maximizes expected total reward. How can we get around this?

This problem did not arise becuase for finite horizon $T$, if rewards have value at most $R$, then our total reward over any trajectory is bounded by $RT$. We can induce a similar bound for the infinite-horizon case by introducing a **discount factor** $\gamma$. This means we multiply the reward term from taking action $a_t$ in state $s_t$ by $\gamma^t$, yielding the objective

$$\max_{\pi}\E_{\tau\sim \mathcal{T}^{\pi}(\tau\mid s)}\Bigg[\sum_{t=0}^\infty \gamma^tr(s_t, a_t)\Bigg]$$

Intuitively, this means we value rewards obtained soon over rewards obtained far into the future. Now, when rewards have value at most $R$, our total reward over any trajectory is bounded by $\frac{R}{1-\gamma}$. (We could normalize the objective by multiplying by $(1 - \gamma)$, but scaling rewards by this constant has no functional effect and introduces extra algebra.)

***Note: This changes the optimal policy.*** Specifically, the optimal policy now takes actions which maximize expected *discounted* total reward. This means we can still replace expectations over the optimal policy with maximums over the action space, as done above: ^0c1458
$$\begin{align*}\E_{\tau \sim \mathcal{T}^*(\tau \mid s)}\Bigg[\sum_{t=0}^\infty \gamma^tr(s_t, a_t)\Bigg] &= \max_{a_0\in \sc A}\;\E_{s_1\sim \mathcal{T}(s_1\mid s, a_0)}\;\max_{a_1\in \sc A}\;\E_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)} \dots\Bigg[\sum_{t=0}^\infty \gamma^tr(s_t, a_t)\Bigg]\\
\E_{\tau \sim \mathcal{T}^*(\tau \mid s, a)}\Bigg[\sum_{t=0}^\infty \gamma^tr(s_t, a_t)\Bigg] &= \E_{s_1\sim \mathcal{T}(s_1\mid s, a)}\;\max_{a_1\in \sc A}\;\E_{s_2\sim \mathcal{T}(s_2\mid s_1, a_1)}\;\max_{a_2\in \sc A} \dots\Bigg[\sum_{t=0}^\infty \gamma^tr(s_t, a_t)\Bigg]\end{align*}$$

#### Infinite-Horizon MDPs: An Alternative Formulation
The discount factor introduced above seems poorly motivated: we introduced it to preserve bounded rewards in infinite-horizon MDPs, but there are many reweightings of reward that could accomplish this (e.g. $t \gamma^{t}$ rather than $\gamma^{t}$).

Here we introduce another way to view the discount factor. Suppose we modify our MDP by introducing a state $d$ known as the death state. We will modify the transition function as follows:
$$\mathcal{T}'(s'\mid s, a) = \gamma \mathcal{T}(s'\mid s, a) + (1 - \gamma) \mathbb{1}(s' = d)$$
That is, every timestep, our new transition function $\mathcal{T}'$ sends us to the same state as $\mathcal{T}$ with probability $\gamma$, but sends us to the death state and ends our rollout with probability $1 - \gamma$. Now note that the MDP is the same as before, but the probability we make it to timestep $t$ is scaled by $\gamma^t$. Thus timestep $t$'s expected contribution to total reward is also scaled by $\gamma^t$, and so

$$\E_{\tau\sim \mathcal{T}'(\tau\mid s)}\Bigg[\sum_{t=0}^\infty r(s_t, a_t)\Bigg] = \E_{\tau\sim \mathcal{T}(\tau\mid s)}\Bigg[\sum_{t=0}^\infty \gamma^tr(s_t, a_t)\Bigg]$$

Thus, introducing a discount factor is equivalent to solving for maximum expected total reward in the modified MDP with the death state. 

We now see we have a choice as to how to interpret the discount factor. We could interpret it as a manual change in objective to provide stability in Infinite-Horizon MDPs; this is called the **Copenhagen interpretation**, after the Copenhagen interpretation of Quantum Physics, as it essentially requires accepting the discount factor without deeper explanation. We could also interpret the discount factor as representing expected total reward in a modified MDP with a death state; this is known as the **death state interpretation**. Surprisingly, these two interpretations of the discount factor have distinct real-world consequences. ^5c76de


