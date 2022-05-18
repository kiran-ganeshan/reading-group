# Soft Actor Critic
## Resources
- [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/sac.html)
- [Original SAC Paper](https://arxiv.org/abs/1801.01290)
- [Updated and Simplified SAC Paper](https://arxiv.org/abs/1812.05905)
- [Original SAC Official Repo, in TF](https://github.com/haarnoja/sac)
- [Current SAC Official Repo, in TF](https://github.com/rail-berkeley/softlearning)
- [Yarats and Kostrikov's PyTorch Implementation](https://github.com/denisyarats/pytorch_sac)
## Introduction
One drawback of standard actor-critic algorithms is that poorly initialized actors can restrict state visitation, preventing the critic from finding high-reward states and encouraging the actor to navigate to these states. [[DQN#^59f907|DQN's epsilon-greedy exploration]] was instrumental in the success of Deep RL in control, and most actor-critic algorithms feature similar hard-coded exploration techniques, e.g. [[DDPG#^d23e5c|DDPG's action noise]].

Imagine if humans had similar exploration techniques: we would explore by making slight mistakes in our actions and observing the results. However, this does not fully capture human exploration: we actively learn to take actions that will teach us more about our environment. We have an intrinsic sense of satisfaction when we learn more about our environment, similar to the extrinsic sense of satisfaction we feel when we accomplish an important task.

In RL, task-based extrinsic satsifcation is modeled by the reward function $r(s, a)$. Could we possibly develop some notion of "intrinsic reward" $r_I(s, a)$ which encourages the agent to explore its environment?

A simple way to quantify how much an agent "explores" is the entropy of the action distribution:
$$r_I(s, a) = H(\pi(\cdot \mid s)) = \mathop{\mathbb{E}}_{a\sim \pi(s)}  \left[-\log\pi(a\mid s)\right]$$
Given an MDP $(S, A, \mathcal{T}, r)$, we can form the **$\alpha$-entropy regularized MDP** $(S, A, \mathcal{T}, r + \alpha r_I)$, where $\alpha$ is a hyperparameter controlling the importance of exploration (similar to $\varepsilon$ in DQN or DDPG).

## Strategy
We will implement an actor-critic algorithm, along with a scheme for adaptively tuning $\alpha$ to ensure that the policy's average entropy remains above a preset minimum $\mathcal{H}$.

Store rollout data in a replay buffer $\mathcal{D}$ and create 
- a Gaussian stochastic policy $\pi_\phi(s) = \mathcal{N}(\mu_\phi(s), \Sigma_\phi(s))$
- a pair of value network $Q_{\theta}(s, a)$
- a target value network $Q_{\theta'}(s, a)$, where $\theta'$ is a moving average of $\theta$

Given a transition $(s, a, r, s', d)\sim\mathcal{D}$, we can train the policy and value networks as follows:
- Update $\theta$ by sampling $a'\sim\pi_\phi(s')$ and then using MSE loss to regress $Q_\theta(s, a)$ to the target
$$y =r + \gamma[Q_{\theta'}(s', a') - \log\pi_\phi(a'\mid s')]$$
- Update $\phi$ by optimizing the following actor loss (utilizing the [[Distribution Modeling with NNs#Differentiation with respect to Samples|reparameterization trick]]):
$$L(\phi) =-Q\Big(s, \;\mu_\phi(s) + \Sigma_\phi(s) \delta\Big) + \log\pi_\phi\Big(\mu_\phi(s) + \Sigma_\phi(s) \delta\;\Big\vert\;  s\Big), \quad \delta\sim\mathcal{N}(0, I)$$
- Update $\alpha$ by optimizing the following temperature loss:
$$L(\alpha) = \alpha(-\log\pi_\phi(a\mid s) - \mathcal{H})$$
where $\mathcal{H}$ is a preset minimum average entropy for the policy.


## Use Cases
Like other actor-critic algorithms, SAC is applicable to
- Discrete and Continuous State Spaces
- Discrete and Continuous Action Spaces
## Algorithm
**SAC**: Soft Actor Critic
**Parameters**: 
- Number training steps $T$
- Replay buffer capacity $N$
- Entropy minimum $\mathcal{H}$
- Target update rate $\rho$
- Learning rates

**Algorithm**:
> Initialize replay buffer $\mathcal{D}$ with capacity $N$\
> Initialize policy NN $\pi_\phi = (\mu_\phi, \Sigma_\phi)$ and value NN $Q_\theta$
> **for** step $\in\{1, \dots, T\}$ **do**:\
> $\qquad$ Retrieve $a= \mu_\phi(s) + \Sigma_\phi(s)\delta$, where $\delta\sim\mathcal{N}(0, I)$\
> $\qquad$ Execute action $a$, retrieve reward $r$ and next state $s'$\
> $\qquad$ Add transition $(s, a, r, s', d)$ with done flag $d$ to $\mathcal{D}$ \
> $\qquad$ Sample transitions and approximate next action: \
> $\qquad\qquad$ Sample transitions $(s_j, a_j, r_j, s_{j+1}, d_j)$ from $\mathcal{D}$\
> $\qquad\qquad$ Sample $\delta_j\sim\mathcal{N}(0, I)$\
> $\qquad\qquad$ Set $a_\phi(s_j) = \mu_\phi(s_j) + \Sigma_\phi(s_j)\delta_j$\
> $\qquad$ Update $\phi$: \
> $\qquad\qquad$ Set $L(\phi) = -\sum_j\left[Q_{\theta}(s_j, a_\phi(s_j)) - \alpha\log\pi_\phi(a_\phi(s_j)\mid s_j)\right]$\
> $\qquad\qquad$ Perform GD on $L$ to update $\phi$ \
> $\qquad$ Update $\theta$: \
> $\qquad\qquad$ Set $q_j = Q_{\theta'}(s_j, a_{\phi}(s_j))-\alpha\log\pi_\phi(a_\phi(s_j)\mid s_j)$\
> $\qquad\qquad$ Set $y_j = r_j +\gamma (1 - d_j)q_{j+1}$\
> $\qquad\qquad$ Set $L(\theta) = \sum_{j} \left(y_j - Q_\theta(s_j,a_j )\right)^2$\
> $\qquad\qquad$ Perform GD on $L$ to update $\theta$\
> $\qquad$ Update $\alpha$:\
> $\qquad\qquad$ Set $L(\alpha) = -\alpha\sum_j [\mathcal{H} + \log\pi(a_j\mid s_j)]$\
> $\qquad\qquad$ Perform GD on $L$ to update $\alpha$\
> $\qquad$ Update $\theta'$:\
> $\qquad \qquad$ $\theta'\leftarrow \rho \theta' + (1-\rho)\theta$
## Analysis
#### Q-Learning
In the $\alpha$-entropy regularized MDP, one might naively assume that since we simply add $\alpha r_I(s, a)$ tot he reward, the Bellman recurrence must become:

$$Q(s, a) = r(s, a) + \alpha r_I(s, a)+ \gamma\mathop{\mathbb{E}}_{s' \sim \mathcal{T}(s, a)}\max_{a'\in A} Q(s', a')$$
However, $Q(s, a)$ inherently holds fixed the first action $a$ taken in the starting state $s$, and our intrinsic reward measures variability in $a$. For this reason, we redefine the Q-function to exclude instrinsic reward from the current timestep:

$$Q_{\text{new}}(s, a) = Q_{\text{old}}(s, a) - \alpha r_I(s, a)$$
and the Bellman recurrence becomes:
$$Q(s, a) = r(s, a) + \gamma\mathop{\mathbb{E}}_{s' \sim \mathcal{T}(s, a)}\max_{a'\in A} [Q(s', a') + \alpha r_I(s', a')]$$

Recall that in DDPG, we approximate $\max_{a'\in A}$ as follows:
$$\max_{a'\in A}Q_\theta(s', a') \approx Q_\theta(s', \pi_{\phi}(s'))$$
where $\pi_{\phi'}$ is the target network of the determinstic policy $\pi_\phi$. However, we need a stochastic policy for entropy regularization to be effective, in which case the approximation becomes
$$\max_{a'\in A}Q_\theta(s', a') \approx \mathop{\mathbb{E}}_{a'\sim\pi_{\phi}(s')}Q_\theta(s', a')$$
this implies
$$\begin{align*}\max_{a'\in A}[Q_\theta(s', a') + \alpha r_I(s', a')] &\approx \mathop{\mathbb{E}}_{a'\sim\pi_{\phi}(s')}\left[Q_\theta(s', a') -  \alpha\mathop{\mathbb{E}}_{a'\sim\pi_\phi(s')} \log\pi(a'\mid s')\right]\\&\approx \mathop{\mathbb{E}}_{a'\sim\pi_\phi(s')}\left[Q_\theta(s', a') -  \alpha \log\pi_\phi(a'\mid s')\right]\end{align*}$$
Thus, the Bellman recurrence becomes
$$Q(s, a) = r(s, a) + \gamma\mathop{\mathbb{E}}_{s' \sim \mathcal{T}(s, a)}\mathop{\mathbb{E}}_{a'\sim\pi_\phi(s')} [Q(s', a') -  \alpha\log\pi_\phi(a'\mid s')]$$
we can approximate the expectation over $s'$ using samples from the replay buffer, and we can approximate the expectation over $a'$ using a Monte-Carlo sample from the current policy (we can't simply save $a'$ in the replay buffer since this would be from an old policy). This gives the Bellman targets
$$y = r(s, a) + \gamma[Q(s', a') - \alpha \log\pi_\phi(a'\mid s')],\quad \text{where }a'\sim\pi_\phi(s')$$
#### Learning a Policy
Recall that we have changed our definition of $Q(s, a)$ for entropy-regularized MDPs to account for the fixed first action $a$:
$$Q_{\text{new}}(s, a) = Q_{\text{old}}(s, a) - \alpha r_I(s, a)$$
In DDPG, the policy objective is $\mathop{\text{argmax}}_\phi Q_{\text{old}}(s, \pi_\phi(s) + \delta)$ where $\delta$ is noise. The objective with a stochastic policy and in terms of the new $Q$ function becomes
$$\begin{align*}\phi^* =&\mathop{\text{argmax}}_\phi Q_{\text{new}}(s, \pi_\phi(s) + \delta) + \alpha r_I(s, a)\\=&\mathop{\text{argmax}}_\phi \mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[Q_{\text{new}}(s, a) - \alpha \mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\log\pi_\phi(a\mid s)\right]\\=&\mathop{\text{argmax}}_\phi \mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[Q(s, a) - \alpha \log\pi_\phi(a\mid s)\right]\end{align*}$$
To turn this into an objective into one we can optimize directly with gradient descent, we utilize the [[Distribution Modeling with NNs#Differentiation with respect to Samples|reparameterization trick]]. We parameterize	a sample from $\pi_\phi$ as follows:
$$a_\phi(s) = \mu_\phi(s) + \Sigma_\phi(s) \delta, \quad \delta\sim\mathcal{N}(0, I)$$
where $\Sigma_\phi(s)$ is diagonal for simplicity. We then solve
$$\phi^* =\mathop{\text{argmax}}_\phi \left[Q(s, a_\phi(s)) - \log\pi_\phi(a_\phi(s)\mid s)\right]$$
This amounts to running gradient descent on $\phi$ with the following loss:
$$\begin{align*}L(\phi) &= -Q\left(s, a_\phi(s)\right) +\alpha \log\pi_\phi(a_\phi(s)\mid s)\\&=-Q\Big(s, \;\mu_\phi(s) + \Sigma_\phi(s) \delta\Big) + \alpha \log\pi_\phi\Big(\mu_\phi(s) + \Sigma_\phi(s) \delta\;\Big\vert\;  s\Big), \quad \delta\sim\mathcal{N}(0, I)\end{align*}$$
where we treat $\delta$ as constant.
#### Learning the Temperature
How can we set the temperature hyperparameter $\alpha$, which determines the scaling of the instrinic entropy reward? The  idea used here is to adaptively set $\alpha$ to keep the policy's average entropy above a preset minimum $\mathcal{H}$:
$$\begin{align*}&\mathop{\mathbb{E}}_{s\sim \mu^\pi(s)}\mathop{\mathbb{E}}_{a\sim \pi_\phi(s)}[-\log\pi_\phi(a\mid s)] \geq \mathcal{H}\\\iff&\mathop{\mathbb{E}}_{s\sim \mu^\pi(s)}\mathop{\mathbb{E}}_{a\sim \pi_\phi(s)}[-\log\pi_\phi(a\mid s) - \mathcal{H}]\geq 0\end{align*}$$
Now consider the optimization objective
$$\argmin_{\alpha} \alpha\mathop{\mathbb{E}}_{s\sim \mu^\pi(s)}\mathop{\mathbb{E}}_{a\sim \pi_\phi(s)} [- \log\pi(a_j\mid s_j)-\mathcal{H}]$$
Suppose we update $\alpha$ using gradient descent on this objective.

When the constraint is satisfied, the solution to the minimization problem is $\alpha=0$, so running gradient descent on this objective will lower $\alpha$, effectively loosening the maximum entropy regularization and allowing the policy to become more deterministic until the constraint becomes unsatisfied (if ever).

When the constraint is unsatisfied, the solution to the minimization problem is $\alpha\to\infty$, so running gradient descent on this objective will raise $\alpha$, effectively strengthening the maximum entropy regularization until the policy meets the constraint.

Note that these GD updates with learning rate $\lambda$ will effectively reduce to
$$\alpha\leftarrow\alpha - \lambda\mathop{\mathbb{E}}_{s\sim \mu^\pi(s)}\mathop{\mathbb{E}}_{a\sim \pi_\phi(s)}[-\log\pi_\phi(a\mid s) - \mathcal{H}]$$
which shows even more clearly that $\alpha$ will increase when the constraint is unsatisfied and decrease otherwise.

## Notes
- The optimization objective for the temperature parameter $\alpha$ is not a hack: it arises directly from considering a constrained optimization objective that maximizes expected total reward subject to the entropy constraint. Because expected total reward depends linearly on the policy, and the constraint is convex in the policy, we can formulate the dual of this constrained optimization. The temperature $\alpha$ arises as the dual variable corresponding to the entropy constraint, and the optimization objective for $\alpha$ comes from the dual optimization problem. Read more about this [in section 5 of the updated SAC paper](https://arxiv.org/pdf/1812.05905.pdf).
- The original SAC paper applies the algorithm to a bounded, continuous action space. In this setting, we use $\tanh$ as a "squashing function" to compress the Gaussian inside the bounded interval $[-1, 1]$:
$$\pi_\phi(s) = \tanh(\mathcal{N}(\mu_\phi(s), \Sigma_\phi(s)))$$
Since $\tanh$ is differentiable we can still use the reparameterization trick:
$$a_\phi(s) = \tanh(\mu_\phi(s) + \Sigma_\phi(s) \delta), \quad \delta\sim\mathcal{N}(0, I)$$
- Note that many other RL algorithms which learn policies do not learn state-dependent policy variances. They instead learn a single set of parameters representing variances for all states. This latter approach works poorly in SAC because of the entropy constraint: in order to satisfy the constraint and still achieve good reward, we must maintain high policy variance in unimportant states (for exploration) while maintaing low policy variance in important states (for high reward).
- The original SAC method trains two value networks $Q$ and $V$ to satisfy
$$V(s) = \mathop{\mathbb{E}}_{a\sim \pi_\phi(s)}[Q(s, a) - \alpha\log\pi_\phi(a\mid s)]$$
$$Q(s, a) = r(s, a) + \gamma \mathop{\mathbb{E}}_{s'\sim\mathcal{T}(s, a)}V(s')$$
- While our policy objective is familiar in light of DDPG, the original SAC paper drew very different inspiration for this objective, deriving it from the following more complex objective:
$$\pi^* = \mathop{\text{argmax}}_\pi \mathop{\mathbb{E}}_{s\sim\mu^\pi} D_{\text{KL}}\left(\pi_\phi(\cdot \mid s)\;\Bigg\vert \Bigg\vert\; \frac{\exp(Q_\theta(s, \cdot))}{Z(s)}\right)$$

