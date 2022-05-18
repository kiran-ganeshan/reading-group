# Advantage Function
## Resources

## Introduction
The [[PG|Policy Gradient]] tells us that when $J(\theta) = V^{\pi_\theta}(s_0)$, we have
$$\nabla_{\theta}J(\theta) \propto \mathop{\mathbb{E}}_{s\sim\mu^{\pi}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi(s)}\Big[Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\Big]$$
This allows us to optimize $J(\theta)$ using a surrogate objective like
$$\mathop{\mathbb{E}}_{s\sim\mu^{\pi}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi(s)}\Big[Q^{\pi}(s, a) \ln\pi_\theta(a\mid s)\Big]$$
where we treat $\pi$ as constant with respect to $\theta$ in the expectations and $Q^\pi$.

How, intuitely, does the maximizing the surrogate objective help us optimize our policy? Notice that $\ln\pi_\theta (a\mid s)$ is always negative, and is more negative when we assign low probability to action $a$ in state $s$. We can view each log-prob $\ln\pi(a\mid s)$ as a penalty for assigning a low probability to action $a$, and these penalties are weighted by the Q-values. Thus, our surrogate objective works by penalizing us *more* for assigning low probabilities to high-value actions. Implicitly, this means penalizing us *less* for assigning high probabilities to high-value actions.

However,  the success of the surrogate objective depends only on differences in $Q(s, a)$ across different actions. If $Q(s, a)$ varies between 95 and 105 in some state $s$, we can impose the same incentives in optimization by subtracting 100 from each $Q$ value and having them vary between -5 and 5. Either way, minimizing the penality involves assigning higher probability (or less negative log-prob) to the higher-valued action.

**Check**: In fact, the latter may be more stable due to the value of the surrogate objective being smaller--large surrogate objectives can lead to large gradients and unstable training.

## Strategy

#### Baslines
We can formalize the intuition above by [[#Policy Gradients are Unaffected by Baselines|showing in our analysis]] that for any **baseline function** $b(s)$ depending only on state:
$$\begin{align*}&\mathop{\mathbb{E}}_{s\sim\mu^{\pi}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi(s)}\Big[Q^{\pi}(s, a)  \nabla_\theta\ln\pi_\theta(a\mid s)\Big]\\=&\mathop{\mathbb{E}}_{s\sim\mu^{\pi}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi(s)}\Big[(Q^{\pi}(s, a) - b(s)) \nabla_\theta\ln\pi_\theta(a\mid s)\Big]\end{align*}$$
This allows us to optimize policies using the **baselined surrogate objective**
$$\mathop{\mathbb{E}}_{s\sim\mu^{\pi}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi(s)}\Big[(Q^{\pi}(s, a) - b(s)) \ln\pi_\theta(a\mid s)\Big]$$
where again $\pi$ is treated as constant in the Q-function and expectations.

In practice, $b(s)$ is useful for eliminating numerical instability resulting from large $Q$-values. The most commonly used baseline is the average of the Q-function over the policy's distribution over actions, or the V-function:
$$V^\pi(s) = \mathop{\mathbb{E}}_{a\sim\pi(s)}Q^\pi(s, a)$$
$$\mathop{\mathbb{E}}_{s\sim\mu^{\pi}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi(s)}\Big[(Q^{\pi}(s, a) - V^{\pi}(s)) \ln\pi_\theta(a\mid s)\Big]$$
This gives rise to the third [[Value Functions|value function]], the **Advantage function**
$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$
which can be thought of measuring the improvement in expected reward when taking action $a$ rather than acting according to the policy.
#### Estimating Advantage
We've found that we can change the Q-Function $Q^\pi(s, a)$ in the policy gradient into the Advantage function $A^\pi(s, a)$, with benefits for stability.

Previously, we estimated $Q^\pi(s, a)$ in the policy gradient using Monte-Carlo samples: simply find the discounted reward sum over the remainder of the trajectory and use these as approximations for $Q^\pi(s, a)$.  This estimator is unbiased, but variance in our estimate in $Q^\pi(s, a)$ is additive accross steps, so for long trajectories this estimator has very high variance.

One idea to reduce this variance is to learn a [[Value Functions|value function]] with a neural network trained using the [[Value Functions#Bellman Recurrence|Bellman Recurrence]]. This only exposes our estimator to reward at the current step, which significantly reduces variance. Directly learning $Q$ is difficult, since the neural network will predict $Q$-values distributed around 0 at initialization. This heavy bias is not realistic for many tasks: recall our theoretical task where $Q$-values are distributed between 95 and 105. This is reasonable in tasks where the agent receives reward each step in a long trajectory.

The Advantage function, however, always has mean zero under the policy's choice of action! If we can somehow build an neural-net based estimator for the advantage function, we can use the baselined surrogate objective to train our policy.

Suppose we gather trajectories $\left\{\left(s^{(i)}_t, a^{(i)}_t, r^{(i)}_t\right)\right\}_{i=1}^N$. We have a few options to develop estimators for the advantage function in the policy gradient.
#### Monte-Carlo Advantage Estimation
One strategy is to estimate $V^\pi$ using a Monte-Carlo estimator based on trajectories, just as we did for $Q^\pi$.  This only works with the [[PG#^921342|highest-variance]] of the policy gradient estimators, which uses the total-trajectory reward $Q^\pi(s_0, a_0)$ as the Q-value estimate for every timestep:
$$\nabla_\theta J(\theta) \approx \nabla_\theta\left[\frac{1}{N}\sum_{i=1}^N\Bigg(\sum_{t=0}^T \gamma^{t} r^{(i)}_t\Bigg)\Bigg(\sum_{t=0}^T \ln\pi_\theta\Big(a_t^{(i)}\mid s_t^{(i)}\Big)\Bigg)\right]$$
We can estimate the value function by averaging total reward across the entire batch of trajectories:
$$\nabla_\theta J(\theta) \approx \nabla_\theta\left[\frac{1}{N}\sum_{i=1}^N\Bigg(\sum_{t=0}^T \gamma^{t} r^{(i)}_t - \frac{1}{N}\sum_{j=1}^N \gamma^tr^{(j)}_t\Bigg)\Bigg(\sum_{t=0}^T \ln\pi_\theta\Big(a_t^{(i)}\mid s_t^{(i)}\Big)\Bigg)\right]$$

#### Advantage Estimation with $V^\pi(s)$
Another estimator, [[#Estimating Advantage with V pi s|derived here]], is based on the following fact:

$$A^{\pi}(s, a)  =\mathop{\mathbb{E}}_{s'\sim\mathcal{T}(s, a)}\Big[\big(r(s, a) + \gamma V^\pi(s') - V^\pi(s)\Big]$$

We can train a neural network $V^\pi_\theta(s)$ to satisfy the   [[Value Functions#Bellman Recurrence|Bellman Recurrence]]:
$$V_\theta^\pi(s) = r(s, a) + \gamma \mathop{\mathbb{E}}_{s'\sim\mathcal{T}(s, a)}V_\theta^\pi(s')$$
which gives rise to the following loss:
$$J(\theta) = \frac{1}{N}\sum_{i=1}^N\sum_{t=1}^{T-1}\left[ r^{(i)}_t +\gamma V^\pi_\theta\left(s^{(i)}_{t+1}\right)-V^\pi_\theta\left(s^{(i)}_t\right)\right]^2  $$
Given the neural network's estimate of the V-function, we can from the baselined surrogate objective for the policy as follows:
$$\nabla_\phi J(\phi)=\nabla_\phi\left[\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^{T-1}\left[r^{(i)}_t +\gamma V^\pi_\theta\left(s^{(i)}_{t+1}\right)-V^\pi_\theta\left(s^{(i)}_t\right)\right]\ln\pi_\phi\left( a^{(i)}_t\;\bigg\vert\; s^{(i)}_t\right)\right]$$

#### Controlling the Bias-Variance Tradeoff
While the Monte-Carlo estimator for advantage us exposes us to the variance of all $T$ steps in each trajectory, the estimator above only exposes us to the variance of 1 step. It removes the remaining variance by estimating return from later steps using $V^\pi(s)$, which introduces bias.  It turns out we can use the [[Value Functions#^24958b|n-step Bellman Recurrence]] to develop an estimator, called the **$n$-step Advantage estimator**, that is exposed to variance from $n$ steps and estimates return from later steps using $V^\pi(s)$. The estimator, [[#Deriving the n -step Advantage Estimator|derived here]], is based on the following fact:

$$A^\pi(s_t, a_t) = \mathop{\mathbb{E}}_{\substack{s_{t+1}\sim\mathcal{T}(s_t, a_t)\\a_{t+1}\sim\pi(s_{t+1})\\ s_{t+2}\sim \mathcal{T}(s_{t+1},a_{t+1})}}\dots\mathop{\mathbb{E}}_{\substack{a_{t+n-1}\sim\pi(s_{t+n-1})\\ s_{t+n}\sim \mathcal{T}(s_{t+n-1},a_{t+n-1})}}\left\{\sum_{t'=t}^{t+n-1} \gamma^{t'-t}r(s_{t'}, a_{t'}) + \gamma^{n} V^\pi(s_{t+n}) - V^\pi(s_t)\right\}$$

so we can calculate the $n$-step Advantage estimator $A_n^\pi(s_t, a_t)$ using Monte-Carlo samples from our trajectories:

$$A^\pi_n(s_t, a_t) = \sum_{t'=t}^{t+n-1} \gamma^{t'-t}r(s_{t'}, a_{t'}) + \gamma^{n} V^\pi(s_{t+n}) - V^\pi(s_t)$$

The $V^\pi$-based estimator from above is the 1-step Advantage estimator, and the Monte-Carlo estimator is the $T$-step Advantage estimator. Generally, larger $n$ implies higher variance and lower bias. The $n$-step Advantage estimator can be used to calculate the surrogate objective as follows:
$$\nabla_\phi J(\phi)=\nabla_\phi\left\{\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^{T-n}\left[\sum_{t'=t}^{t+n-1}\gamma^{t'}r^{(i)}_{t'} +\gamma^n V^\pi_\theta\left(s^{(i)}_{t+n}\right)-V^\pi_\theta\left(s^{(i)}_t\right)\right]\ln\pi_\phi\left( a^{(i)}_t\;\bigg\vert\; s^{(i)}_t\right)\right\}$$

#### Generalized Advantage Estimation
Rather than choosing a single value of $n$ and using the $n$-step Advantage estimator
$$A^\pi_n(s_t, a_t) = \sum_{t'=t}^{t+n-1}\gamma^{t'}r(s_{t'}, a_{t'}) + \gamma^nV^\pi(s_{t+n})-V^\pi(s_t)$$
we could also create a weighted combination of these estimators to blend high-bias and high-variance information:
$$A^\pi(s, a)\approx \sum_{n=1}^\infty w_nA^\pi_n(s, a),\qquad\text{where}\quad\sum_{n=1}^\infty w_n=1$$
one convenient choice for these weights is a weighting scheme which places exponentially less weight on higher variance estimates: $w_n = (1-\lambda)\lambda^{n-1}$. This gives rise to the **Generalized Advantage Estimator with parameter $\lambda$**, or GAE($\lambda$):
$$A^\pi_{\text{GAE}, \lambda}(s, a)= \sum_{n=1}^\infty (1-\lambda)\lambda^{n-1}A^\pi_n(s, a)$$
It turns out there is a more convenient way to calculate  GAE($\lambda$), [[#Deriving Generalized Advantage Estimation|derived here]]:
$$A^\pi_{\text{GAE},\lambda}(s_t, a_t) =\sum_{t'=t}^\infty (\lambda\gamma)^{t'-t}\delta_{t'},\qquad\text{where}\quad \delta_t = r(s_t, a_t) + \gamma V^\pi(s_{t+1})-V^\pi(s_t)$$
We can use GAE($\lambda$) to optimize our policy with the following surrogate objective:

$$\delta^{(i)}_t = r\left(s^{(i)}_t, a^{(i)}_t\right) + \gamma V^\pi\left(s^{(i)}_{t+1}\right)-V^\pi\left(s^{(i)}_t\right)$$
$$\nabla_\phi J(\phi)=\nabla_\phi\left\{\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^{T}\left[\sum_{t'=t}^{T-1}(\lambda\gamma)^{t'-t}\delta^{(i)}_{t'}\right]\ln\pi_\phi\left( a^{(i)}_t\;\bigg\vert\; s^{(i)}_t\right)\right\}$$

## Analysis
#### Policy Gradients are Unaffected by Baselines
To see why we can introduce a baseline to the policy gradient (without introducing bias), note that
$$\begin{align*}\mathop{\mathbb{E}}_{a\sim\pi(s)} \Big[b(s)\nabla_\theta\ln\pi(a\mid s)\Big]&=\int_A b(s)\pi_\theta(a\mid s)\nabla_\theta\ln\pi_\theta(a\mid s)\;da\\&=b(s)\int_A \pi_\theta(a\mid s)\nabla_\theta\ln\pi_\theta(a\mid s)\;da\\&=b(s)\int_A\nabla_\theta\pi_\theta(a\mid s)\;da\\&=b(s)\nabla_\theta\left[\int_A\pi_\theta(a\mid s)\;da\right]=0\end{align*}$$
since $\int_A\pi_\theta(a\mid s)\;da = 1$ for any policy, and the gradient of a constant is 0. By linearity of expectation, this allows us to state
$$\begin{align*}&\mathop{\mathbb{E}}_{a\sim\pi(s)} \Big[(Q^\pi(s, a) - b(s))\nabla_\theta\ln\pi(a\mid s)\Big]\\=&\mathop{\mathbb{E}}_{a\sim\pi(s)} \Big[Q^\pi(s, a)\nabla_\theta\ln\pi(a\mid s)\Big] - \mathop{\mathbb{E}}_{a\sim\pi(s)} \Big[b(s)\nabla_\theta\ln\pi(a\mid s)\Big]\\=&\mathop{\mathbb{E}}_{a\sim\pi(s)} \Big[Q^\pi(s, a)\nabla_\theta\ln\pi(a\mid s)\Big]\end{align*}$$
Taking the expectation of both sides with respect to $s\sim\mu^\pi(s\mid s_0)$ gives the desired result.
#### Estimating Advantage with $V^\pi(s)$
Can we find an estimator with lower variance using a neural network? Note that 
$$Q^\pi(s, a) = r(s, a) + \gamma\mathop{\mathbb{E}}_{s'\sim\mathcal{T}(s, a)}\big[V^\pi(s')\big]$$
This means
$$\begin{align*}A^\pi(s, a) &= r(s, a) + \gamma\mathop{\mathbb{E}}_{s'\sim\mathcal{T}(s, a)}\big[V^\pi(s')\big] - V^\pi(s)\\&= \mathop{\mathbb{E}}_{s'\sim\mathcal{T}(s, a)}\big[r(s, a) + \gamma V^\pi(s') - V^\pi(s)\big]\end{align*}$$
Thus we can write the baselined surrogate objective as
$$\begin{align*}&\mathop{\mathbb{E}}_{s\sim\mu^{\pi}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi(s)}\Big[A^{\pi}(s, a)  \ln\pi(a\mid s)\Big]\\=&\mathop{\mathbb{E}}_{s\sim\mu^{\pi}(s\mid s_0)}\;
\mathop{\mathbb{E}}_{a\sim\pi(s)}\Big[\mathop{\mathbb{E}}_{s'\sim\mathcal{T}(s, a)}\big[r(s, a) + \gamma V^\pi(s') - V^\pi(s)\big]  \ln\pi(a\mid s)\Big]\\=&\mathop{\mathbb{E}}_{s\sim\mu^{\pi}(s\mid s_0)}\;
\mathop{\mathbb{E}}_{a\sim\pi(s)}\mathop{\mathbb{E}}_{s'\sim\mathcal{T}(s, a)}\Big[\big(r(s, a) + \gamma V^\pi(s') - V^\pi(s)\big)  \ln\pi(a\mid s)\Big]\end{align*}$$
Note also that for timestep $t$ in trajectory $i$, we have $$\begin{align*}s^{(i)}_t&\sim\mu^\pi(s\mid s_0)\\ a^{(i)}_t &\sim\pi\left(s^{(i)}_t\right)\\ s^{(i)}_{t+1}&\sim\mathcal{T}\left(s^{(i)}_{t+1}\;\bigg\vert\;s^{(i)}_{t}, a^{(i)}_{t}\right)\end{align*}$$
This allows us to approximate the expectations using Monte-Carlo samples from our trajectories:

#### Deriving the $n$-step Advantage Estimator
We start with the 1-step Advantage estimator:
$$A^\pi(s_t, a_t) = \mathop{\mathbb{E}}_{s_{t+1}\sim\mathcal{T}(s_t, a_t)}\big[r(s_t, a_t) + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)\big]$$
and we apply the $(n-1)$[[Value Functions#^24958b|-step Bellman Recurrence]] to the second term $\gamma V^\pi(s_{t+1})$:

$$V^\pi(s_{t+1}) = \mathop{\mathbb{E}}_{\substack{a_{t+1}\sim\pi(s_{t+1})\\ s_{t+2}\sim \mathcal{T}(s_{t+1},a_{t+1})}}\dots\mathop{\mathbb{E}}_{\substack{a_{t+n-1}\sim\pi(s_{t+n-1})\\ s_{t+n}\sim \mathcal{T}(s_{t+n-1},a_{t+n-1})}} \left[\sum_{t'=t+1}^{t+n-1} \gamma^{t'-t-1}r(s_{t'}, a_{t'}) + \gamma^{n-1} V^\pi(s_{t+n})\right]$$
This yields
$$\begin{align*}A^\pi(s_t, a_t) &= \mathop{\mathbb{E}}_{\substack{s_{t+1}\sim\mathcal{T}(s_t, a_t)\\a_{t+1}\sim\pi(s_{t+1})\\ s_{t+2}\sim \mathcal{T}(s_{t+1},a_{t+1})}}\dots\mathop{\mathbb{E}}_{\substack{a_{t+n-1}\sim\pi(s_{t+n-1})\\ s_{t+n}\sim \mathcal{T}(s_{t+n-1},a_{t+n-1})}}\left\{r(s_t, a_t) + \gamma\left[ \sum_{t'=t+1}^{t+n-1} \gamma^{t'-t-1}r(s_{t'}, a_{t'}) + \gamma^{n-1} V^\pi(s_{t+n})\right] - V^\pi(s_t)\right\}\\&= \mathop{\mathbb{E}}_{\substack{s_{t+1}\sim\mathcal{T}(s_t, a_t)\\a_{t+1}\sim\pi(s_{t+1})\\ s_{t+2}\sim \mathcal{T}(s_{t+1},a_{t+1})}}\dots\mathop{\mathbb{E}}_{\substack{a_{t+n-1}\sim\pi(s_{t+n-1})\\ s_{t+n}\sim \mathcal{T}(s_{t+n-1},a_{t+n-1})}}\left\{\sum_{t'=t}^{t+n-1} \gamma^{t'-t}r(s_{t'}, a_{t'}) + \gamma^{n} V^\pi(s_{t+n}) - V^\pi(s_t)\right\}\end{align*}$$
Putting this back into the baselined surrogate objective, we get
$$\begin{align*}&\mathop{\mathbb{E}}_{s\sim\mu^{\pi}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi(s)}\Big[A^{\pi}(s, a)  \ln\pi(a\mid s)\Big]\\=&\mathop{\mathbb{E}}_{\substack{s_t\sim\mu^{\pi}(s)\\ a_t\sim\pi(s_t)\\ s_{t+1}\sim\mathcal{T}(s_t, a_t)}}\;
\dots\mathop{\mathbb{E}}_{\substack{ a_{t+n-1}\sim\pi(s_{t+n-1})\\s_{t+n}\sim\mathcal{T}(s_{t+n-1}, a_{t+n-1})}}\;\left\{\left[\sum_{t'=t}^{t+n-1}\gamma^{t'}r(s_{t'}, a_{t'}) + \gamma^n V^\pi(s_{t+n}) - V^\pi(s_t)\right]  \ln\pi(a_t\mid s_t)\right\}\end{align*}$$
The trajectories we collect during rollouts contain Monte-Carlo samples we can use to approximate the expectations, and we can again approximate $V^\pi(s)$ using a neural net trained via the [[Value Functions#Bellman Recurrence|Bellman Recurrence]]. This allows us to calculate the surrogate objective as
$$\nabla_\phi J(\phi)=\nabla_\phi\left\{\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^{T-n}\left[\sum_{t'=t}^{t+n-1}\gamma^{t'}r^{(i)}_{t'} +\gamma^n V^\pi_\theta\left(s^{(i)}_{t+n}\right)-V^\pi_\theta\left(s^{(i)}_t\right)\right]\ln\pi_\phi\left( a^{(i)}_t\;\bigg\vert\; s^{(i)}_t\right)\right\}$$

#### Deriving Generalized Advantage Estimation
We will derive the following method of calculating the GAE:
$$A^\pi_{\text{GAE}}(s_t, a_t) =\sum_{t'=t}^\infty (\lambda\gamma)^{t'-t}\delta_{t'},\qquad\text{where}\quad \delta_t = r(s_t, a_t) + \gamma V^\pi(s_{t+1})-V^\pi(s_t)$$
We start by noting that, with $\delta_t$ defined as above, we have $r(s_t, a_t) + \gamma V^\pi(s_{t+1}) = \delta_t + V^\pi(s_t)$, which means that
$$\begin{align*}A^\pi_2(s_t, a_t) &= r(s_t, a_t) + \gamma r(s_{t+1}, a_{t+1}) + \gamma^2V^\pi(s_{t+2})-V^\pi(s_t)\\&= r(s_t, a_t) + \gamma \big[r(s_{t+1}, a_{t+1}) + \gamma V^\pi(s_{t+2})\big]-V^\pi(s_t)\\&= r(s_t, a_t) + \gamma \big[\delta_{t+1} + V^\pi(s_{t+1})\big]-V^\pi(s_t)\\&= r(s_t, a_t) + \gamma V^\pi(s_{t+1})-V^\pi(s_t) + \gamma \delta_{t+1} \\&= \delta_t + \gamma \delta_{t+1} \end{align*}$$
More generally, we have
$$\begin{align*}A^\pi_n(s_t, a_t) &= \sum_{t'=t}^{t+n-1}\gamma^{t'}r(s_{t'}, a_{t'})  + \gamma^nV^\pi(s_{t+n})-V^\pi(s_t)\\&= \sum_{t'=t}^{t+n-2}\gamma^{t'}r(s_{t'}, a_{t'}) +\gamma^{t+n-1}r(s_{t+n-1}, a_{t+n-1}) + \gamma^nV^\pi(s_{t+n})-V^\pi(s_t)\\&= \sum_{t'=t}^{t+n-2}\gamma^{t'}r(s_{t'}, a_{t'}) +\gamma^{t+n-1}(r(s_{t+n-1}, a_{t+n-1}) + \gamma V^\pi(s_{t+n}))-V^\pi(s_t)\\&= \sum_{t'=t}^{t+n-2}\gamma^{t'}r(s_{t'}, a_{t'}) +\gamma^{t+n-1}(\delta_{t+n-1} + V^\pi(s_{t+n-1}))-V^\pi(s_t)\\&=\gamma^{t+n-1}\delta_{t+n-1}+ \sum_{t'=t}^{t+n-2}\gamma^{t'}r(s_{t'}, a_{t'}) + \gamma^{t+n-1} V^\pi(s_{t+n-1})-V^\pi(s_t)\\&=\gamma^{t+n-1}\delta_{t+n-1}+ A^\pi_{n-1}(s_t, a_t)\end{align*}$$
Together with the base case $\delta_t = A_1^\pi(s_t, a_t)$, this proves by induction that
$$A^\pi_n(s_t, a_t) = \sum_{t'=t}^{t+n-1}\gamma^{t'-t}\delta_{t'}$$
With this fact, we simply must switch the order of the sums in our definition of the GAE to obtain the desired result. The following notation will help:
$$\mathbb{1}(\text{statement}) = \begin{cases}1&\text{if statement is true}\\0&\text{otherwise}\end{cases}$$
Putting it all together, we have
$$\begin{align*}A^\pi_{\text{GAE}}(s_t, a_t)&= \sum_{n=1}^\infty (1-\lambda)\lambda^{n-1}\sum_{t'=t}^{t+n-1}\gamma^{t'-t}\delta_{t'}\\&=\sum_{n=0}^\infty (1-\lambda)\lambda^n\sum_{t'=t}^{t+n}\gamma^{t'-t}\delta_{t'}\\&=(1-\lambda)\sum_{n=0}^\infty \sum_{t'=t}^{t+n}\lambda^n\gamma^{t'-t}\delta_{t'}\\&=(1-\lambda)\sum_{n=0}^\infty \sum_{t'=t}^\infty\mathbb{1}(t'\leq t+n)\lambda^n\gamma^{t'-t}\delta_{t'}\\&=(1-\lambda)\sum_{t'=t}^\infty\sum_{n=0}^\infty \mathbb{1}(n\geq t'-t)\lambda^n\gamma^{t'-t}\delta_{t'}\\&=(1-\lambda)\sum_{t'=t}^\infty\sum_{n=t'-t}^\infty \lambda^n\gamma^{t'-t}\delta_{t'}\\&=(1-\lambda)\sum_{t'=t}^\infty\sum_{n=0}^\infty \lambda^{t' - t +n}\gamma^{t'-t}\delta_{t'}\\&=(1-\lambda)\sum_{t'=t}^\infty (\lambda\gamma)^{t'-t}\delta_{t'}\sum_{n=0}^\infty \lambda^n\\&=(1-\lambda)\sum_{t'=t}^\infty (\lambda\gamma)^{t'-t}\delta_{t'}\left(\frac{1}{1-\lambda}\right)\\&=\sum_{t'=t}^\infty (\lambda\gamma)^{t'-t}\delta_{t'}\end{align*}$$