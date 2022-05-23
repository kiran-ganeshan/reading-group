# Batch Constrained Q-Learning
Requirements: [[RL Paradigms]], [[DQN]], [[VAE]]
## Introduction
We've seen that DQN is an **off-policy** algorithm, allowing it to train on data gathered by an arbitrary policy. This also means we can (in principle) train a DQN **offline**, or on a fixed dataset of trajectories.

However, offline DQNs suffer from **extrapolation error**, in which the NN predicts unrealistic values for $Q(s, a)$ when the state-action pair $(s, a)$ is not seen in the dataset. This error can be transferred to the state-action pairs seen in the dataset via the Bellman Recurrence:
$$Q(s, a) = r(s, a) + \gamma \E_{s'\sim\mathcal{T}(s'\mid s, a)} \max_{a'\in A} Q(s', a')$$
What if there are actions $a'$ such that no state-action pair in the dataset is close to $(s', a')$? Then, $Q$ will suffer from extrapolation error on $(s', a')$ for each of these actions $a'$, meaning it will very likely overestimate $\max_{a'\in A} Q(s', a')$. When we train the DQN on this overestimation, the exptrapolation error from $(s', a')$ leaks into $(s, a)$.

One way to avoid extrapolation error on a dataset $\mathcal{D}$ is to restrict the maximum in the Bellman Recurrence to be over state-actions present in $\mathcal{D}$. To formalize this, we simply alter the Bellman Recurrence:
$$Q(s, a) = r(s, a) + \gamma \E_{s'\sim\mathcal{T}(s'\mid s, a)} \max_{\substack{a'\in A\\(s', a')\in \mathcal{D}}} Q(s', a')$$
where $(s', a')\in\mathcal{D}$ is a condition on $a'\in A$. How can we tractably train on this altered Bellman Recurrence?

## Strategy

A simple approach involves learning a state-conditioned generative model $G(a\mid s)$ such that all $a\sim G(s)$ satisfies our condition $(s, a)\in\mathcal{D}$. This simplifies the Bellman Recurrence to
$$Q(s, a) = r(s, a) + \gamma \E_{s'\sim\mathcal{T}(s'\mid s, a)} \max_{a'\sim G(s')} Q(s', a')$$
We can tractably evaluate the maximum by sampling $N$ actions $a_i'\sim G(s')$ each step, and then evaluating $\max_{i\in[N]}Q(s', a'_i)$. BCQ uses a [[VAE]] as the state-conditioned generative model.

## Use Cases
BCQ is applicable to 
- Discrete and Continuous State Spaces
- Discrete and Continuous Action Spaces

Notice that introducing the generative model allows us to easily generalize [[DQN]] to continuous action spaces.

## Algorithm
**BCQ**: Batch Constrained Q-Learning

**Parameters**: 
- Number of total steps $T$
- Replay Buffer capacity $N$
- Exploration parameter $\ep$
- Target update rate $\tau$
- VAE latent space size $d$
- VAE learning rate
- Q-function learning rate

**Algorithm**:
> Initialize replay buffer $\mathcal{D}$ with capacity $N$\
> Initialize $Q$-function neural network with random weights $\theta$\
> Initialize target network with random weights $\theta'=\theta$\
> Initialize VAE $G_\phi = (E_\phi, D_\phi)$ \
> **for** $(s, a, s', r)\in\mathcal{D}$ **do**:\
> $\qquad$ $\mu_\phi, \sigma_\phi = E_\phi(s, a), \quad z\sim\mathcal{N}(\mu,\sigma), \quad a_\phi' = D_\phi(s, z)$\
> $\qquad$ Set $L(\phi) = (a - a_\phi')^2 + D_{KL}(\mathcal{N}(0, 1)\vert\vert\mathcal{N}(\mu_\phi, \sigma_\phi))$ \
> $\qquad$ Perform gradient descent on $L$ to update $\phi$ \
> $\qquad$ **for** $i\in[N]$ **do**:\
> $\qquad\qquad$ Sample $z_i'\sim\mathcal{N}(0, 1)$\
> $\qquad\qquad$ Set $a_i'\leftarrow D_\phi(s', z_i')$\
> $\qquad$ Set $y = r + \gamma (1-d) \max_{i\in [n]}Q_{\theta'}(s', a_i')$\
> $\qquad$ Set our loss to $L(\theta) =  \left(y - Q_\theta(s,a )\right)^2$\
> $\qquad$ Perform gradient descent on $L$ to update $\theta$ \
> $\qquad$ Update target network: $\theta' \leftarrow (1-\tau)\theta' + \tau\theta$

## Miscellaneous
- The original BCQ algorithm included a perturbation network $\xi_\psi(s, a)$ designed to enable access to low-probability actions in the outskirts of state coverage, which would otherwise require many samples from the generative model to consider. These perturbations were then clipped to $[-\Phi, \Phi]$ for a hyperparmeter $\Phi$, yielding the Bellman Recurrence
$$Q(s, a) = r(s, a) + \gamma \E_{s'\sim\mathcal{T}(s'\mid s, a)} \max_{a'\sim G(s')} Q(s', a' + )$$
- The original BCQ algorithm also utilized a variant of [[TD3#Clipped Double Q-Learning]] to prevent overestimation bias: $$Q_{\theta_j}(s, a) =r + \gamma\left[(1-\lambda)\min_{i\in\{1, 2\}}\max_{k\in[N]}Q_{\theta_i'}(s', a'_k) + \lambda \max_{i\in\{1, 2\}}\max_{k\in[N]}Q_{\theta_i'}(s', a'_k)\right]$$ This variant removes an amount $\lambda$ of weight from the minimum of the independently biased Q estimates, and places this weight on the maximum. This prevents underestimation in the offline setting.