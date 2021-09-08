# Markov Decision Processes
Requirements: 

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

## Formal Definition

More concretely, An MDP is a tuple $(S, A, \mathcal{T}, r)$, meaning it consists of 
1. A **State Space** $S$, the set of all of the states in which the agent could possibly be
2. An **Action Space** $A$, the set of all the actions the agent could possibly take (sometimes a function of state)
3. A **Transition Operator** $\mathcal{T} : S\times A \to \mathcal{D}(S)$, which defines the distribution $\mathcal{T}(s' \mid s, a)$ over possible next states $s'$ given the current state $s$ and the action $a$ just taken
4. A **Reward Function** $r: S\times A\to \mathbb{R}$, which determines the reward $r(s, a)$ received by the agent when it takes action $a$ in state $s$

Note that the reward function only depends on the current state-action pair $(s_t, a_t)$, and not the previous states or actions in our current trajectory. This is known as the **Markov Property** and is what distinguishes a regular old Decision Process from an MDP.

## Trajectories
A possible solution to an MDP is represented by a **policy** $\pi: S\to \mathcal{D}(A)$. Given a policy, we let $\mathcal{T}^{\pi}$ refer to the resulting distribution over trajectories:
$$\mathcal{T}^{\pi}(\tau \mid s_0) = \prod_{k=0}^{\infty}\pi(a_k\mid s_k) \mathcal{T}(s_{k+1}\mid s_k, a_k)$$
$$\mathcal{T}^{\pi}(\tau \mid s_0, a_0) = \prod_{k=0}^{\infty} \mathcal{T}(s_{k+1}\mid s_k, a_k)\pi(a_{k+1}\mid s_{k+1})$$
We let $\mathcal{T}^*$ refer to the distribution over trajectories under the optimal policy:
$$\mathcal{T}^*(\tau \mid s_0) = \prod_{k=0}^{\infty}\pi^*(a_k\mid s_k) \mathcal{T}(s_{k+1}\mid s_k, a_k)$$
$$\mathcal{T}^*(\tau \mid s_0, a_0) = \prod_{k=0}^{\infty} \mathcal{T}(s_{k+1}\mid s_k, a_k)\pi^(a_{k+1}\mid s_{k+1})$$