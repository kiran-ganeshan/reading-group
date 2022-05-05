

# Policy Gradient
Requirements: [[Value Functions]], [[Neural Network]]

## Resources
- [Sutton & Barto 13.1 - 13.4](http://incompleteideas.net/book/RLbook2020.pdf) 
- Credits for proof sketch: [Lil'Log](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
- [Towards Data Science](https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d)

## Introduction
Recall that $V^{\pi}(s)$ is the expected reward-to-go in state $s$ under policy $\pi$. This is precisely what we'd like to maximize when solving an MDP! 

Thanks to a result known as the policy gradient theorem, we can directly parameterize the policy as $\pi_{\theta}$ and optimize via gradient descent with respect to $\theta$. 

This note describes how this works. We first choose the objective $J(\theta) = V^{\pi}(s_0)$ where $s_0$ is our starting state, meaning we hope to optimize expected total rewards under our policy starting in state $s_0$.

## Strategy
The policy gradient theorem tells us
$$\nabla_{\theta}J(\theta) \propto \E_{s\sim\mu^{\pi}(s\mid s_0)}\; 
\E_{a\sim\pi(s)}\Big[Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\Big]$$
where $\mu^{\pi}(s\mid s_0)$ is the distribution of state visits from state $s_0$ under policy $\pi$. We would like to generate rollouts with our policy and use this theorem to directly update its parameters based on the rollouts we see. To do this, we must figure out
1. [[Distribution Modeling with NNs|Parameterize]] $\pi_\theta(a\mid s)$ 
2. Approximate $Q^{\pi_\theta}$
3. Approximate the expectations

Approximating expectations is simple: the states $s$ seen during trajectories are distributed according to $\mu^{\pi}(s\mid s_0)$, and the actions $a$ are distributed according to $\pi(a\mid s)$. Hence sampling state-action pairs from trajectories provides a Monte Carlo approximation of the expectations.

Representing $\pi_\theta$ is merely a matter of standard distribution [[Distribution Modeling with NNs|parameterization]], with the additional detail that we use the surrogate loss $L(s, a, \theta) = Q^{\pi}(s, a)\ln \pi_\theta(a\mid s)$ as our loss function (as opposed to something like the negative-log-likelihood loss).

Approximating $Q^{\pi_\theta}$ involves more in-depth design choices, which we explore below. The upshot is that we will use one of the following approximations, which differ only in application of discounts:
$$\nabla_\theta J(\theta) \approx \nabla_\theta\left[\frac{1}{N}\sum_{i=1}^N\Bigg(\sum_{t=0}^T \gamma^{t} r^{(i)}_t\Bigg)\Bigg(\sum_{t=0}^T \ln\pi_\theta\Big(a_t^{(i)}\mid s_t^{(i)}\Big)\Bigg)\right]$$

$$\nabla_\theta J(\theta) \approx \nabla_\theta\left[\frac{1}{N}\sum_{i=1}^N\sum_{t=0}^T\Bigg(\sum_{t'=t}^T \gamma^{t' - t} r^{(i)}_{t'}\Bigg) \ln\pi_\theta\Big(a_t^{(i)}\mid s_t^{(i)}\Big)\right]$$
$$\nabla_\theta J(\theta) \approx \nabla_\theta\left[\frac{1}{N}\sum_{i=1}^N\sum_{t=0}^T\Bigg(\sum_{t'=t}^T \gamma^{t'} r^{(i)}_{t'}\Bigg) \ln\pi_\theta\Big(a_t^{(i)}\mid s_t^{(i)}\Big)\right]$$

where $\left(s^{(j)}_t, a^{(j)}_t, r^{(j)}_t\right)$ is the transition in episode $j$ at time step $t$.

## Use Cases

The policy gradient is applicable to
- Discrete and Continuous State Spaces
- Discrete and Continuous Action Spaces
	- Note: in infinite action spaces, we cannot parameterize arbitrary action distributions, so we must make choices about which family of action distributions (e.g. Gaussian, Exponential) to represent.

## Algorithm

**PG**: Policy Gradient
**Parameters**: 
- Number of episodes $N$
- Horizon $T$
- Timesteps until training $T_0$
- Parameterization $\xi$ (see [[PG#^df0b0f|above]])
- $Q$-value estimator (see [[PG#^df0b0f|above]])

**Algorithm**:
> Initialize neural network $f$ with random weights $\theta$\
> Parameterize the policy as described above: $\pi_\theta(a\mid s) = \xi(a\mid f_\theta(s))$\
> **for** episode $\in\{1, \dots, N\}$ **do**:\
> $\qquad$ **for** $t\in\{1, \dots, T\}$ **do**:\
> $\qquad\qquad$ **if** $N T + t < T_0$ (fewer than $T_0$ timesteps have passed): \
> $\qquad\qquad\qquad$ $a_t\set$ random action\
> $\qquad\qquad$ **else**:\
> $\qquad\qquad\qquad$ $a_t\sim \pi(a_t\mid s_t)$\
> $\qquad\qquad$ Execute action $a_t$, retrieve reward $r_t$ and next state $s_{t+1}$\
> $\qquad$ Calculate $Q$-value estimates $\hat Q(s_j, a_j)$ for each transition\
> $\qquad$ Set our loss to $L = \sum_{j} \hat Q(s_j, a_j)\ln\pi_\theta(a_j\mid s_j)$\
> $\qquad$ Perform gradient descent on $L$ to update $\theta$

## Analysis
We prove the following theorem showing the correctness of the policy gradient:

**Theorem**: For a given starting state $s_0$, let $J(\theta) = V^{\pi_\theta}(s_0)$. Then
$$\nabla_{\theta}J(\theta) \propto \E_{s\sim\mu^{\pi}(s\mid s_0)}\; 
\E_{a\sim\pi(s)}\Big[Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\Big]$$
where $\propto$ is a symbol meaning "proportional to" (we will derive constants of proportionality in the proof) and $\mu^{\pi}(s\mid s_0)$ is the distribution of states visited by $\pi$ starting in state $s_0$. This theorem is important because 
1. We can sample $s\sim\mu^{\pi}(s\mid s_0)$ by running the policy starting with $s_0$ and sampling states from the resultant trajectories, allowing us to approximate the outer expectation. 
2. Given a state $s$ sampled from the trajectories, the action taken at $s$ in the trajectory was sampled from $\pi(s)$, so if we choose $a$ to be this action than effectively $a\sim\pi(s)$, allowing us to approximate the inner expectation.

Together, these facts show that if run the policy $\pi$ in the environment to generate trajectories, and then we sample state-action pairs from trajectories, we can approximate the two expectations, allowing us to evaluate this expression.

**Proof**: Note that for any starting state $s$, by the [[Value Functions#Bellman Recurrence | Bellman Recurrence Relation]] relating $V^{\pi}(s)$ to an expectation of $V^{\pi}(s')$ over the next state $s'$, we have
$$\nabla_{\theta}V^{\pi}(s) = \nabla_\theta \E_{a\sim \pi_\theta(a\mid s)}\bigg\{r(s, a) + \gamma\E_{s'\sim \mathcal{T}(s' \mid s,a)}\;V^{\pi}(s')\bigg\}$$

We might be tempted to try to move the gradient through the expectation, but not so fast! The expectation is over $\pi_\theta$, which depends on $\theta$! We must expand the expectation, and so the proof splits depending on whether we are in a discrete or continuous space. We will present the discrete case, and to obtain the continuous case one simply replaces sums over $S$ and $A$ with integrals over these continuous spaces.

In the discrete case, the expectation becomes a sum over the action space: 
$$\begin{align*}

\nabla_{\theta}V^{\pi}(s) &= \nabla_\theta \sum_{a\in A} \pi_\theta(a\mid s)\bigg\{r(s, a) + \gamma\E_{s'\sim \mathcal{T}( s,a)}\;V^{\pi}(s')\bigg\}\\

&=\sum_{a\in A} \Bigg\{\bigg[r(s, a) + \gamma\E_{s'\sim\mathcal{T}( s, a)}V^{\pi}(s')\bigg]\nabla_\theta\pi_{\theta}(a\mid s) + \pi_\theta(a\mid s)\nabla_\theta\bigg[r(s, a) + \gamma\E_{s'\sim\mathcal{T}( s, a)}V^\pi (s')\bigg]\Bigg\}\\

&=\sum_{a\in A} \Bigg\{\bigg[r(s, a) + \gamma\E_{s'\sim\mathcal{T}( s, a)}V^{\pi}(s')\bigg]\nabla_\theta\pi_{\theta}(a\mid s) + \pi_\theta(a\mid s) \gamma\E_{s'\sim\mathcal{T}( s, a)}\nabla_\theta V^\pi (s')\Bigg\}\\

\end{align*}$$

We now recognize the term in brackets as the [[Value Functions#^4e628e | partial recurrence]] of $Q^{\pi}$, so we can substitute $Q^{\pi}$ to obtain
$$\begin{align*}\nabla_\theta V^{\pi}(s) &= \sum_{a\in A} \bigg\{Q^{\pi}(s, a)\nabla_\theta\pi_{\theta}(a\mid s) + \pi_\theta(a\mid s) \gamma\E_{s'\sim\mathcal{T}( s, a)}\nabla_\theta V^\pi (s')\bigg\}\\
&= \sum_{a\in A} Q^{\pi}(s, a)\nabla_\theta\pi_{\theta}(a\mid s) + \sum_{a\in A}\pi_\theta(a\mid s) \gamma\E_{s'\sim\mathcal{T}( s, a)}\nabla_\theta V^\pi (s')\end{align*}$$

This is a recurrence relation on $\nabla_\theta V^{\pi}$. The goal of the policy gradient is to to use this recurrence relation to approximate the gradient. In order to accmplish this, we're going to unroll the recurrence relation and obtain an expression for the policy gradient based on sample trajectories. 

Note that to approximate this gradient, we can either run the policy to generate infinitely long training trajectories (called the **continuing case**), or we can run the policy to generate training trajectories of horizon $T$ (called the **episodic case**). 

In the episodic case, we set $\gamma=1$, and we use the finite-horizon Bellman Recurrence, which has a base case in which the expectation over future $V$ disappears when $s$ is the last state in the trajectory. Therefore we only unroll to depth $T$.

In the continuing case, we set $0 << \gamma < 1$, and we use the infinite-horizon Bellman Recurrence, which has no base case, so we unroll to infinite depth and our policy gradient will depend on an infinite sum.

For reasons we will explain later, we will only present the episodic case. However, the continuing case is very similar, and will be left as an exercise to the reader. To make deriving the continuing case easier, we will continue to include discounts until we actually unroll the recurrence, at which point we will explain why we are only deriving the episodic case, set $\gamma=1$, and unroll only to depth $T$.

But first, a few tricks to help unroll:
1. We will define
$$\phi(s) =\sum_{a\in A}Q^{\pi}(s, a)\nabla_\theta\pi_\theta(a\mid s)$$
for convenience.
Using this, our recurrence relation now looks like
$$\begin{align*}
\nabla_\theta V^{\pi}(s) &= \phi(s) + \sum_{a\in A}\pi_\theta(a\mid s) \gamma\E_{s'\sim\mathcal{T}( s, a)}\nabla_\theta V^\pi (s')
\end{align*}$$

2. We will write the expectation over the transition operator in its weighted sum  form:
$$\E_{s'\sim\mathcal{T}(s'\mid s, a)}f(s') = \sum_{s'\in S}\mathcal{T}(s'\mid s, a)f(s')$$
The recurrence relation now becomes
$$\begin{align*}
\nabla_\theta V^{\pi}(s) &= \phi(s) + \sum_{a\in A}\pi_\theta(a\mid s) \gamma\sum_{s'\in S}\mathcal{T}(s'\mid s, a)\nabla_\theta V^\pi (s')\\
&= \phi(s) + \gamma\sum_{s'\in S}\Bigg[\sum_{a\in A}\pi_\theta(a\mid s) \mathcal{T}(s'\mid s, a)\Bigg]\nabla_\theta V^\pi (s')\\
\end{align*}$$
where we can introduce the brackets since $\nabla_\theta V^{\pi}(s')$ does not depend on $a$. This allows us another liberty.
3. We introduce the following notation for the total probability of transitioning from $s$ to $s'$ under $\pi$:
$$P_\pi (s\to s') = \sum_{a\in A}\pi(a\mid s)\mathcal{T}(s'\mid s, a)$$
So that
$$\nabla_\theta V^{\pi}(s)= \phi(s) + \gamma\sum_{s'\in S}P_\pi(s\to s')\nabla_\theta V^\pi (s')$$
We also introduce the following notation for the total probability of transition from s to s' on a trajectory of length $k$ under $\pi$:
$$\begin{align*}P^0(s\to s') &= \begin{cases}1&\text{if } s=s'\\0&\text{if }s\neq s'\end{cases}\\P^1_{\pi}(s\to s') &= P(s\to s')\\
P^2_{\pi}(s\to s'') &= \sum_{s'\in S}P(s\to s')P(s'\to s'')\\P_{\pi}^3(s\to s''')&=\sum_{s''\in S}P^2(s\to s'')P(s''\to s''')\\P_{\pi}^k(s_0\to s_k)&=\sum_{s_{k-1}\in S}P^{k-1}(s_0\to s_{k-1})P(s_{k-1}\to s_k)\end{align*}$$
This may seem like a lot of notation. I promise it will become useful.
4. We enumerate the states with numbers ($s_0, s_1, s_2, \dots$) rather than primes ($s, s', s'', \dots$), as we have in the last equation above.

With our new representation of the recurrence, it is much easier to unroll it:
$$\begin{align*}

\nabla_\theta V^{\pi}(s_0) &= \phi(s_0) + \gamma\sum_{s_1\in S}P_\pi(s_0\to s_1)\nabla_\theta V^\pi (s_1)\\

&= \phi(s_0) + \gamma\sum_{s_1\in S}P_\pi(s_0\to s_1) \Bigg[\phi(s_1) + \gamma\sum_{s_2\in S}P_\pi(s_1\to s_2)\nabla_\theta V^\pi (s_2)\Bigg]\\

&= \phi(s_0) + \gamma\sum_{s_1\in S}P_\pi(s_0\to s_1) \phi(s_1) + \gamma^2\sum_{s_1\in S}P_\pi(s_0\to s_1)\sum_{s_2\in S}P_\pi(s_1\to s_2)\nabla_\theta V^\pi (s_2)\\

&= \phi(s_0) + \gamma\sum_{s_1\in S}P_\pi(s_0\to s_1) \phi(s_1) + \gamma^2\sum_{s_2\in S}\Bigg[\sum_{s_1\in S}P_\pi(s_0\to s_1)P_\pi(s_1\to s_2)\Bigg]\nabla_\theta V^\pi (s_2)\\

&= \phi(s_0) + \gamma\sum_{s_1\in S}P_\pi(s_0\to s_1) \phi(s_1) + \gamma^2\sum_{s_2\in S}P_{\pi}^2(s_0\to s_2)\nabla_\theta V^\pi (s_2)\\

\end{align*}$$

We can unroll it one more time to make a point, this time exaggerating the pattern in the terms in our presentation:
$$\begin{align*}

\nabla_\theta V^{\pi}(s_0) =\quad & \phi(s_0) \\&+ \gamma\sum_{s_1\in S}P_\pi(s_0\to s_1) \phi(s_1) \\&+ \gamma^2\sum_{s_2\in S}P_{\pi}^2(s_0\to s_2)\Bigg[\phi(s_2) + \gamma\sum_{s_3\in S}P_\pi(s_2\to s_3)\nabla_\theta V^\pi (s_3)\Bigg]\\

=\quad & \phi(s_0) \\&+ \gamma\sum_{s_1\in S}P_\pi(s_0\to s_1) \phi(s_1) \\&+ \gamma^2\sum_{s_2\in S}P_{\pi}^2(s_0\to s_2)\phi(s_2) \\&+ \gamma^3\sum_{s_2\in S}P_{\pi}^2(s_0\to s_2)\sum_{s_3\in S}P_\pi(s_2\to s_3)\nabla_\theta V^\pi (s_3)\\

=\quad & \phi(s_0) \\&+ \gamma\sum_{s_1\in S}P_\pi(s_0\to s_1) \phi(s_1) \\&+ \gamma^2\sum_{s_2\in S}P_{\pi}^2(s_0\to s_2)\phi(s_2) \\&+ \gamma^3\sum_{s_3\in S}\Bigg[\sum_{s_2\in S}P_{\pi}^2(s_0\to s_2)P_\pi(s_2\to s_3)\Bigg]\nabla_\theta V^\pi (s_3)\\

=\quad & \phi(s_0) \\&+ \gamma\sum_{s_1\in S}P_\pi(s_0\to s_1) \phi(s_1) \\&+ \gamma^2\sum_{s_2\in S}P_{\pi}^2(s_0\to s_2)\phi(s_2) \\&+ \gamma^3\sum_{s_3\in S}P_{\pi}^3(s_0\to s_3)\nabla_\theta V^\pi (s_3)\\

\end{align*}$$

Now we must decide how far to unroll (either to horizon $T$ or to $\infty$) depending on whether we are in the continuing or episodic case. In practice, we normally train policy gradient algorithms using rollouts with a fixed horizon $T$, so the episodic case is more frequently used in practice. Therefore we will present it and leave the continuing case as an exercise to the reader.

### Episodic Case

We set $\gamma = 1$ and unroll to depth $T$. Note we have
$$\phi(s_0) = \sum_{s\in S}P^0_{\pi}(s_0\to s)\phi(s)$$
We'll plug this into our unrolled recurrence, and we'll rename $s_1, s_2, \dots, s_T$ to $s$ since they are now summed over in separate terms, yielding
$$\begin{align*}

\nabla_\theta V^{\pi}(s_0) &= \phi(s_0) + \sum_{s_1\in S}P_\pi(s_0\to s_1) \phi(s_1)  + \dots + \sum_{s_T\in S}P_{\pi}^T(s_0\to s_T)\phi(s_T)\\

&=\sum_{s\in S}P^0_{\pi}(s_0\to s)\phi(s) + \sum_{s\in S}P_\pi(s_0\to s) \phi(s)  + \dots+ \sum_{s\in S}P_{\pi}^T(s_0\to s)\phi(s)\\

&=\sum_{s\in S}\Bigg[\sum_{k=0}^T P^k_{\pi}(s_0\to s)\Bigg]\phi(s) \\

\end{align*}$$
This prompts us to define
$$\eta^{\pi}(s\mid s_0) = \sum_{k=0}^T P^k_{\pi}(s_0\to s)$$
so that
$$\nabla_\theta V^{\pi}(s_0) = \sum_{s\in S}\eta^{\pi}(s\mid s_0)\phi(s)$$
$\eta(s\mid s_0)$ is the expected visit count of $s$ when starting in $s_0$ (without any discount). 

In this form, we are close to our final solution, which involves an outer expectation over states and an inner expectation over actions. Note also that the expression above includes an outer sum over states and an inner sum over actions (the inner sum is implicit in $\phi$). We will transform the inner sum over actions (in $\phi$) into an expectation later; first let's turn the outer sum over states into an expectation. 

Note this would be trivial if $\eta^{\pi}$ was a distribution, but it is an expected visit count. We can, however, normalize it into a distribution. The resulting distribution, $\mu^{\pi}$, will represent the distribution of total state visits over the trajectory:
$$\begin{align*}\mu^{\pi}(s\mid s_0) &= \frac{\eta^{\pi}(s\mid s_0)}{\underset{s\in S}{\sum}\eta^{\pi}(s\mid s_0)}\\&=\frac{\eta^{\pi}(s\mid s_0)}{\underset{s\in S}{\sum}\overset{T}{\underset{k\in\mathbb{N}}{\sum}}P^k_{\pi}(s_0\to s)}\\&=\frac{\eta^{\pi}(s\mid s_0)}{\overset{T}{\underset{k\in\mathbb{N}}{\sum}}\underset{s\in S}{\sum}P^k_{\pi}(s_0\to s)}\end{align*}$$
Note that $\underset{s\in S}{\sum}P^k_{\pi}(s_0\to s) = 1$ since we visit exactly 1 state at step $k$, and as a result we have
$$\mu^{\pi}(s\mid s_0) = \frac{\eta^{\pi}(s\mid s_0)}{\overset{T}{\underset{k=0}{\sum}}1}=\frac{1}{T}\eta^{\pi}(s\mid s_0) $$
$$\Longrightarrow \eta^{\pi}(s\mid s_0) = T\mu^{\pi}(s\mid s_0)$$
Now that we have normalized $\eta^{\pi}$, we can plug it back into the policy gradient expression:
$$\begin{align*}\nabla_\theta J(\theta) &= \nabla_\theta V^{\pi}(s_0) \\ &=\sum_{s\in S}\eta^{\pi}(s\mid s_0)\phi(s)\\&= T\sum_{s\in S}\mu^{\pi}(s\mid s_0)\phi(s)\\&=T\E_{s\sim \mu^{\pi}(s\mid s_0)}\phi(s)\end{align*}$$
This gives us the outer expectation over the distribution $\mu^\pi$ of state visits. (Recall this is helpful because this is precisely the distribution of states when we randomly sample from trajectories.) We still need to obtain the inner expectation over the policy, and we can use the following clever trick:
$$\begin{align*}

\phi(s) &= \sum_{a\in A}Q^{\pi}(s, a)\nabla_\theta\pi_\theta(a\mid s)\\

&= \sum_{a\in A}Q^{\pi}(s, a)\pi_\theta(a\mid s)\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}\\

&= \sum_{a\in A}\pi_\theta(a\mid s)Q^{\pi}(s, a)\nabla_\theta\ln\pi_\theta(a\mid s)\\

&= \E_{a\sim\pi(a\mid s)}Q^{\pi}(s, a)\nabla_\theta\ln\pi_\theta(a\mid s)\\


\end{align*}$$
where we multiply and divide by $\pi$ to bring back the expectation. Putting this all together we have
$$\nabla_\theta J(\theta) = T\E_{s\sim \mu^{\pi}(s\mid s_0)}\;\E_{a\sim\pi(a\mid s)}Q^{\pi}(s, a)\nabla_\theta\ln\pi_\theta(a\mid s)$$
which is our intended result with a constant of proportionality $T$.
### Continuing Case
This is left as an exercise to the reader. Step through the proof as in the episodic case, but including discounts. What is the new constant of proportionality? How can we interpret $\eta^\pi$ and $\mu^\pi$ in this case?
##### Continuing Case Solution
This time, we constrain $\gamma \in (0, 1)$ and we unroll to infinite depth. Note we have
$$\phi(s_0) = \gamma^0\sum_{s\in S}P^0_{\pi}(s_0\to s)\phi(s)$$
We'll plug this into our unrolled recurrence, and we'll rename $s_1, s_2, \dots, s_T$ to $s$ since they are now summed over in separate terms, yielding
$$\begin{align*}

\nabla_\theta V^{\pi}(s_0) &= \phi(s_0) + \gamma\sum_{s_1\in S}P_\pi(s_0\to s_1) \phi(s_1) + \gamma^2\sum_{s_2\in S}P_{\pi}^2(s_0\to s_2)\phi(s_2) + \dots\\

&=\gamma^0\sum_{s\in S}P^0_{\pi}(s_0\to s)\phi(s) + \gamma^1\sum_{s\in S}P_\pi(s_0\to s) \phi(s)  + \gamma^2\sum_{s_2\in S}P_{\pi}^2(s_0\to s)\phi(s) + \dots\\

&=\sum_{s\in S}\Bigg[\sum_{k=0}^\infty \gamma^kP^k_{\pi}(s_0\to s)\Bigg]\phi(s) \\

\end{align*}$$
This prompts us to define
$$\eta^{\pi}(s\mid s_0) = \sum_{k=0}^\infty \gamma^kP^k_{\pi}(s_0\to s)$$
so that
$$\nabla_\theta V^{\pi}(s_0) = \sum_{s\in S}\eta^{\pi}(s\mid s_0)\phi(s)$$
We can interpret $\eta(s\mid s_0)$ as the expected *discounted* visit count of $s$ when starting in $s_0$, where visits that occur after $k$ timesteps are discounted by a factor of $\gamma^k$. 

This has a natural interpretation if we modify our MDP by introducing a *death state*, and give the agent a probability $(1 - \gamma)$ of dying and being sent to this state at any timestep. Note then that the discount applied to rewards at step $k$ is simply $\gamma^k$,  the probability we survive to $k$ steps at all. Hence expected total reward in this modified MDP is simply discounted expected total reward in our original MDP. Likewise, expected visit count in the modified MDP is simply discounted expected visit count in the original MDP. So we can reinterpret the discounted objective as expected total rewards, and we can reinterpret $\eta^{\pi}(s\mid s_0)$ as an expected visit count, both in a modified MDP. With this interpretation, we proceed as before.

As before, we proceed by normalizing $\eta^{\pi}$ into a distribution $\mu^{\pi}$ so that we can turn our sum over states into an expectation:
$$\begin{align*}\mu^{\pi}(s\mid s_0) &= \frac{\eta^{\pi}(s\mid s_0)}{\underset{s\in S}{\sum}\eta^{\pi}(s\mid s_0)}\\&=\frac{\eta^{\pi}(s\mid s_0)}{\underset{s\in S}{\sum}\overset{\infty}{\underset{k\in\mathbb{N}}{\sum}}\gamma^kP^k_{\pi}(s_0\to s)}\\&=\frac{\eta^{\pi}(s\mid s_0)}{\overset{\infty}{\underset{k\in\mathbb{N}}{\sum}}\gamma^k\underset{s\in S}{\sum}P^k_{\pi}(s_0\to s)}\end{align*}$$
Again, note that $\underset{s\in S}{\sum}P^k_{\pi}(s_0\to s) = 1$, so
$$\mu^{\pi}(s\mid s_0) = \frac{\eta^{\pi}(s\mid s_0)}{\overset{\infty}{\underset{k=0}{\sum}}\gamma^k}=(1 - \gamma)\eta^{\pi}(s\mid s_0) $$
$$\Longrightarrow \eta^{\pi}(s\mid s_0) = \frac{\mu^{\pi}(s\mid s_0)}{1 - \gamma}$$
$$\begin{align*}\nabla_\theta J(\theta) &= \nabla_\theta V^{\pi}(s_0) \\ &=\sum_{s\in S}\eta^{\pi}(s\mid s_0)\phi(s)\\&= \frac{1}{1-\gamma}\sum_{s\in S}\mu^{\pi}(s\mid s_0)\phi(s)\\&=\frac{1}{1-\gamma}\E_{s\sim \mu^{\pi}(s\mid s_0)}\phi(s)\end{align*}$$
This time, our constant of proportionality is $\frac{1}{1 - \gamma}$. We proceed with the trick to turn the inner sum over actions into an expectation, bringing us to the final solution:
$$\begin{align*}

\phi(s) &= \sum_{a\in A}Q^{\pi}(s, a)\nabla_\theta\pi_\theta(a\mid s)\\

&= \sum_{a\in A}Q^{\pi}(s, a)\pi_\theta(a\mid s)\frac{\nabla_\theta\pi_\theta(a\mid s)}{\pi_\theta(a\mid s)}\\

&= \sum_{a\in A}\pi_\theta(a\mid s)Q^{\pi}(s, a)\nabla_\theta\ln\pi_\theta(a\mid s)\\

&= \E_{a\sim\pi(a\mid s)}Q^{\pi}(s, a)\nabla_\theta\ln\pi_\theta(a\mid s)\\


\end{align*}$$
$$\nabla_\theta J(\theta) = \frac{1}{1-\gamma}\E_{s\sim \mu^{\pi}(s\mid s_0)}\;\E_{a\sim\pi(a\mid s)}Q^{\pi}(s, a)\nabla_\theta\ln\pi_\theta(a\mid s)$$

## Details: Approximating $Q^{\pi_\theta}$

To approximate $Q^{\pi_\theta}$, we note that the total reward over a trajectory is a sample from the distribution of possible total rewards. We want the expected value of this distribution, and the Weak Law of Large Numbers tells us that if we collect enough samples, the average of our samples should converge the expected value of the distribution of total reward. This means we can use the total discounted reward over a trajectory containing $(s, a)$ as a Monte-Carlo estimate of the $Q$-value. This, however, raises two questions.
1. Should we use total-trajectory reward as our estimated $Q$-value, or should we use reward-to-go?
2. If we use reward-to-go, should we apply the discount factor relative to the current transition, or relative to the beginning of the trajectory?  

Differing answers to these questions give rise to three esimators for $Q^{\pi_\theta}$. Suppose we've gathered $N$ rollouts 
$$\tau^{(i)} = \Big(\Big(s_0^{(i)}, a_0^{(i)}\Big), \Big(s_1^{(i)}, a_1^{(i)}\Big), \dots, \Big(s_T^{(i)}, a_T^{(i)}\Big)\Big)$$
We would like to esimate $Q^{\pi}$ and plug this estimate into our gradient.
- If we use total-trajectory reward:
$$\begin{align*}\nabla_\theta J(\theta) &\approx \frac{1}{N}\sum_{i=1}^N\sum_{t=0}^T\Bigg(\sum_{t'=0}^T \gamma^{t'} r\Big(s_{t'}^{(i)}, a_{t'}^{(i)}\Big)\Bigg)\Bigg( \nabla_\theta\ln\pi_\theta\Big(a_t^{(i)}\mid s_t^{(i)}\Big)\Bigg)\\&\approx \frac{1}{N}\sum_{i=1}^N\Bigg(\sum_{t=0}^T \gamma^{t} r\Big(s_{t}^{(i)}, a_{t}^{(i)}\Big)\Bigg)\Bigg(\sum_{t=0}^T \nabla_\theta\ln\pi_\theta\Big(a_t^{(i)}\mid s_t^{(i)}\Big)\Bigg)\end{align*}$$
- If we use reward-to-go and apply the discount factor relative to the current transition:
$$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^N\sum_{t=0}^T\Bigg(\sum_{t'=t}^T \gamma^{t' - t} r\Big(s_{t'}^{(i)}, a_{t'}^{(i)}\Big)\Bigg) \nabla_\theta\ln\pi_\theta\Big(a_t^{(i)}\mid s_t^{(i)}\Big)$$
- If we use reward-to-go and apply the discount factor relative to the beginning of the trajectory:
$$\begin{align*}\nabla_\theta J(\theta) &\approx \frac{1}{N}\sum_{i=1}^N\sum_{t=0}^T\Bigg(\sum_{t'=t}^T \gamma^{t'} r\Big(s_{t'}^{(i)}, a_{t'}^{(i)}\Big)\Bigg)\Bigg( \nabla_\theta\ln\pi_\theta\Big(a_t^{(i)}\mid s_t^{(i)}\Big)\Bigg)\\&\approx \frac{1}{N}\sum_{i=1}^N\sum_{t=0}^T\gamma^t\Bigg(\sum_{t'=t}^T \gamma^{t' - t} r\Big(s_{t'}^{(i)}, a_{t'}^{(i)}\Big)\Bigg) \nabla_\theta\ln\pi_\theta\Big(a_t^{(i)}\mid s_t^{(i)}\Big)\end{align*}$$

Notice that the third approach is just like the second approach, but with transitions at timestep $t$ downweighted by $\gamma^t$. Remember that $\gamma^t$ is the probability we reach timestep $t$ in the [[MDP#Infinite-Horizon MDPs An Alternative Formulation | death state interpretation]] of the discount factor.

Therefore, the second approach is an estimator for the $Q$-value at a transition in the [[MDP#^5c76de| Cophenhagen interpretation]] of the discount factor. In this interpretation, all transitions are weighted equally. The third approach corresponds to the [[MDP#^5c76de| death state interpretation]] of the discount factor, in which later transitions are downweighted by $\gamma^t$ to account for the probability that we never reach these transitions at all, due to premature death.






