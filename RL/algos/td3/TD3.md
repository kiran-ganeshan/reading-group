# Twin Delayed DDPG
Requirements: [[DDPG]], [[Importance Sampling for PG]]
## Resources
- [Original Paper](https://arxiv.org/pdf/1802.09477.pdf)
- [Spinning up](https://spinningup.openai.com/en/latest/algorithms/td3.html#)
## Introduction
[[DDPG]] utilizes advances in Q-learning with neural networks for continuous control. Experiments show DDPG suffers from **overestimation bias**, in which the Q-network tends to overestimate the value of states, causing the policy to exploit the Q-network's mistakes. How can we make adjustments to combat overestimation bias?
## Strategy
Twin Delayed Deep Deterministic Policy Gradient (TD3) is a variant of DDPG that makes use of three additional tricks to improve performance:
- **Delayed Policy Updates**: Experiments show that updating the policy frequently causes larger variances in Q-value estimates, so we only update the policy once per $k$ critic updates.
- **Clipped Double Q-Learning**: [[Double Q-Learning]] has been shown to improve performance. A generalization of [[Double Q-Learning#Modern Double Q-Learning|modern Double Q-Learning]] could be used with an actor-critic algorithm: $$Q(s, a) = r + \gamma Q_{\theta'}\left(s', \pi_\phi(s')\right)$$ in practice, however, when we update $\pi_\phi$ more slowly than $Q_\theta$, $\pi_\phi(s)$ is closer to maximizing the target network $Q_{\theta'}$ than the current network $Q_\theta$, defeating the point of double Q-learning. One might then consider using the [[Double Q-Learning#Original Double Q-Learning|original Double Q-Learning]] strategy: $$\begin{align*}Q_{\theta_1}(s, a) &=r + \gamma Q_{\theta_2'}(s', \pi_{\phi_1}(s'))\\Q_{\theta_2}(s, a) &=r + \gamma Q_{\theta_1'}(s', \pi_{\phi_2}(s'))\end{align*}$$ where $\pi_{\phi_1}$ is trained to maximize $Q_{\theta_1}$, and likewise for $\pi_{\phi_2}$. However, this simplification of the original Double Q-learning technique turns out to show better results, while only storing one set of policy parameters $\phi$: $$\begin{align*}Q_{\theta_1}(s, a) &=r + \gamma \min_{i\in\{1, 2\}}Q_{\theta_i'}(s', \pi_{\phi}(s'))\\Q_{\theta_2}(s, a) &=r + \gamma \min_{i\in\{1, 2\}}Q_{\theta_i'}(s', \pi_{\phi}(s'))\end{align*}$$ This approach is called Clipped Double Q-Learning, and it helps combat overestimation by taking the minimum of two independently biased estimates of Q when constructing targets.
- **Target Policy Smoothing**: Lastly, despite the above optimizations which reduce actor-critic interdependence and improve Q-function performance, our actor still may overfit to local maxima in the Q-function. To prevent this, we use a target policy network $\pi_{\phi'}$ to generate actions in the Bellman target, along with Gaussian noise clipped to $[-\Delta, \Delta]$ for some $\Delta\in \mathbb{R}$:$$Q_{\theta_j}(s, a) =r + \gamma \min_{i\in\{1, 2\}}Q_{\theta_i'}\left(s', \pi_{\phi'}(s') + \operatorname{clip}(\delta, -\Delta, \Delta)\right), \quad \delta\sim\mathcal{N}(0, \varepsilon)$$Note there will be two hyperparameters $\varepsilon$, one from original DDPG which controls noise in actions generated during training rollouts, and the one introduced here which controls noise in actions used in the Bellman target.

## Use Cases
As with DDPG, TD3 is applicable to
- Discrete and Continuous State Spaces
- Discrete and Continuous Action Spaces

## Algorithm
**TD3**: Twin Delayed Deep Deterministic Policy Gradient
**Parameters**: 
- Number training steps $T$
- Replay buffer capacity $N$
- Rollout action noise parameter $\varepsilon_a$
- Target smoothing noise parameter $\varepsilon_t$
- Target noise cap $\Delta$
- Learning rates for gradient descent
- Target update rate $\rho$

**Algorithm**:
> Initialize replay buffer $\mathcal{D}$ with capacity $N$\
> Initialize policy NN $\pi_\phi$ and value NN $Q^*_\theta$
> **for** step $\in\{1, \dots, T\}$ **do**:\
> $\qquad$ Retrieve $a= \pi_\phi(s) + \delta$, where $\delta\sim\mathcal{N}(0, \varepsilon)$\
> $\qquad$ Execute action $a$, retrieve reward $r$ and next state $s'$\
> $\qquad$ Sample transitions $(s_j, a_j, r_j, s_{j+1}, d_j)$ from $\mathcal{D}$\
> $\qquad$ If $\text{step}\equiv 0 \pmod k$, update $\phi$, $\theta'_i$, and $\phi'$:
> $\qquad\qquad$ Set $L(\phi) = -\sum_jQ_{\theta}(s_j, \pi_\phi(s_j))$
> $\qquad\qquad$ Perofrm GD on $L$ to update $\phi$
> $\qquad \qquad$ $\theta_i'\leftarrow \rho \theta_i' + (1-\rho)\theta_i\quad$ for $i\in\{1, 2\}$
> $\qquad \qquad$ $\phi'\leftarrow \rho \phi' + (1-\rho)\phi$
> $\qquad$ Update $\theta_i$:
> $\qquad\qquad$ Set $\delta_j = \operatorname{clip}(\sigma_j, -\Delta, \Delta)$ where $\sigma_j\sim\mathcal{N}(0, \varepsilon_t)$\
> $\qquad\qquad$ Set $y_j = r_j +\gamma (1 - d_j)\min_iQ_{\theta'_i}(s_{j+1}, \pi_{\phi'}(s_{j+1}) + \delta_j)$\
> $\qquad\qquad$ Set $L(\theta_i) = \sum_{j} \left(y_j - Q_{\theta_i}(s_j,a_j )\right)^2$\
> $\qquad\qquad$ Perform GD on $L$ to update $\theta_1$ and $\theta_2$
