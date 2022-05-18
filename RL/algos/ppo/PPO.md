# Proximal Policy Optimization
Requirements: [[TRPO]], [[Advantage Function]]

## Resources
- [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html#references)
- [Original PPO Paper](https://arxiv.org/abs/1707.06347)
## Introduction

In [[TRPO]], we impose a KL constraint on the policy optimization process to maintain high similarity between the state visitation distributions of the current policy $\pi_\theta$ and the prior policy $\pi_{\theta_k}$ that generated training data. This allowed us to ignore the ratio between state visitation probabilities in the importance-weighted policy gradient, which simplified the surrogate objective:

$$J(\theta) = \mathop{\mathbb{E}}_{s\sim\mu^{\pi_{\theta_k}}(s\mid s_0)}
\mathop{\mathbb{E}}_{a\sim\pi_{\theta_k}(s)}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_k}(a\mid s)}A^{\pi_{\theta_k}}(s, a) \right]$$

Recall that the Vanilla PG loss works by penalizing the policy for placing low probability on actions with high $Q$-value via a $Q$-weighted sum of log-probabilities. This surrogate objective has an even simpler interpretation: it encourages the policy to increase the probability of high-Advantage actions and decrease the probability of low-Advantage actions.

This leads to a simpler way to constrain the current policy $\pi_\theta$'s KL-Divergence with the old policy $\pi_{\theta_k}$: simply clip the importance weights in the surrogate objective. If $A^{\pi_{\theta_k}}(s, a)$ is positive, clip the importance weight above at $1 + \varepsilon$. That way, when the importance weight rises above this value, meaning $\pi_\theta$ is growing too different from $\pi_{\theta_k}$, we begin to replace the importance weight with the constant $1 + \varepsilon$, and $\pi_\theta$ stops receiving gradient signal. Similarly, if $A^{\pi_{\theta_k}}(s, a)$ is negative, clip the importance weight below at $1 - \varepsilon$.

## Strategy

As mentioned above, we clip the importance weight so that it stays below $1 + \varepsilon$ when $A^{\pi_{\theta_k}}(s, a)  > 0$, and we clip the importance weight so that it stays above $1-\varepsilon$ when $A^{\pi_{\theta_k}}(s, a)  < 0$. Given trajectories $\left\{\left(s^{(i)}_t, a^{(i)}_t, r^{(i)}_t\right)\right\}_{i=1}^N$,  we can calculate the PPO surrogate objective as follows:
1. Calculate and store original probabilities: $\pi^{(i)}_t = \pi_{\theta_k}\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)$
2. Calculate Advantage estimates: $\hat A^{(i)}_t\approx A^{\pi_{\theta_k}}\left(s^{(i)}_t, a^{(i)}_t\right)$ 
3. Calculate the surrogate objective:
$$\begin{align*}J(\theta) &= \frac{1}{NT}\sum_{i=1}^n\sum_{t=1}^T\begin{cases}\min\left(\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi^{(i)}_t}, 1 + \varepsilon\right)\hat A^{(i)}_t &\text{ if }\hat A^{(i)}_t>0\\\max\left(\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi^{(i)}_t}, 1 - \varepsilon\right)\hat A^{(i)}_t&\text{ otherwise}\end{cases}\\&= \frac{1}{NT}\sum_{i=1}^n\sum_{t=1}^T\min\left\{\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi^{(i)}_t}\hat A^{(i)}_t,\; \text{clip}_{1-\varepsilon}^{1+\varepsilon}\left(\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi^{(i)}_t}\right)\hat A^{(i)}_t\right\}\end{align*}$$
where $\text{clip}_a^b(x)$ simply clips $x$ to the interval $[a, b]$.
## Use Cases
PPO is applicable to

-   Discrete and Continuous State Spaces
-   Discrete and Continuous Action Spaces
## Algorithm
**PPO**: Proximal Policy Optimization
**Parameters**: 
- Maximum episode length $T$
- Number of episodes per iteration $N$
- Number of iterations $K$
- Importance Sample Clip Value $\varepsilon$
- [[Advantage Function#Estimating Advantage|Advantage Function Estimator]] $\mathop{\text{AdvEst}}(s,  r, V^\pi)$

**Algorithm**:
> Initialize policy NN $\pi_\theta$ and value NN $V^\pi_\phi$
> **for** epoch $k\in\{1, \dots, K\}$ **do**:\
> $\qquad$ **for** episode $i\in\{1, \dots, N\}$ **do**:\
> $\qquad\qquad$ **for** step $t\in\{1, \dots, T\}$ **do**:\
> $\qquad\qquad\qquad$ Take action $a_t^{(i)}\sim \pi_{\theta_k}\left(s^{(i)}\right)$\
> $\qquad\qquad\qquad$ Get reward $r_t^{(i)}$, state $s_{t+1}^{(i)}$\
> $\qquad\qquad\qquad$ Store logprob $\pi_t^{(i)} = \pi_{\theta_k}\left(a_t^{(i)}\bigg\vert s_t^{(i)}\right)$\
> $\qquad$ Estimate advantage $\hat A^{(i)}_t = \mathop{\text{AdvEst}}\left(s_t^{(i)},  r_t^{(i)}, V^\pi_\phi\right)$ \
> $\qquad$ $J(\theta) = \frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\min\left\{\frac{\pi_\theta\left(a^{(i)}_t\big\vert s^{(i)}_t\right)}{\pi^{(i)}_t}\hat A^{(i)}_t,\; \text{clip}_{1-\varepsilon}^{1+\varepsilon}\left(\frac{\pi_\theta\left(a^{(i)}_t\big\vert s^{(i)}_t\right)}{\pi^{(i)}_t}\right)\hat A^{(i)}_t\right\}$ \
> $\qquad$ Run GD on $J(\theta)$ to update $\theta$ \
> $\qquad$ $L(\phi) = \frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T \left(r_t^{(i)} + \gamma V^\pi_\phi\left(s_{t+1}^{(i)}\right)-V^\pi_\phi\left(s_t^{(i)}\right)\right)^2$ \
> $\qquad$ Run GD on $L(\phi)$ to update $\phi$

## Analysis
We provided two expressions for the PPO surrogate objective: one that was justifiable intuitively, and another that was convenient to calculate. Here we prove that these are equal, starting with the easy-to-calculate form:
$$\min\left\{\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi_{\theta_k}\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}\hat A^{(i)}_t,\; \text{clip}_{1-\varepsilon}^{1+\varepsilon}\left(\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi_{\theta_k}\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}\right)\hat A^{(i)}_t\right\}$$
We break this up into two cases:
- The advantage function is positive: $\hat A^{(i)}_t >0$
In this case the surrogate objective becomes
$$\begin{align*}&\min\left\{\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi_{\theta_k}\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)},\; \text{clip}_{1-\varepsilon}^{1+\varepsilon}\left(\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi_{\theta_k}\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}\right)\right\}\hat A^{(i)}_t\\=&\min\left\{\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi_{\theta_k}\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)},\; 1 + \varepsilon\right\}\hat A^{(i)}_t\end{align*}$$
- The advantage function is negative: $\hat A^{(i)}_t <0$
In this case the surrogate objective becomes
$$\begin{align*}&\max\left\{\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi_{\theta_k}\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)},\; \text{clip}_{1-\varepsilon}^{1+\varepsilon}\left(\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi_{\theta_k}\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}\right)\right\}\hat A^{(i)}_t\\=&\max\left\{\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi_{\theta_k}\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)},\; 1 - \varepsilon\right\}\hat A^{(i)}_t\end{align*}$$

Putting this all together, we recieve exactly the form of the surrogate objective we justified intuitively:



$$J(\theta) = \begin{cases}\min\left(\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi_{\theta_k}\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}, 1 + \varepsilon\right)\hat A^{(i)}_t &\text{ if }\hat A^{(i)}_t>0\\\max\left(\frac{\pi_\theta\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}{\pi_{\theta_k}\left(a^{(i)}_t\Big\vert s^{(i)}_t\right)}, 1 - \varepsilon\right)\hat A^{(i)}_t&\text{ otherwise}\end{cases}$$