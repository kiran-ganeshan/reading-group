# Natural Policy Gradient
Requirements: [[PG]], [[Conjugate Gradient Method]]
## Resources

## Introduction

Recall that we can interpret the gradient descent step
$$\theta_{t+1} \leftarrow \theta_t - \alpha \nabla_\theta J(\theta_t)$$
as the solution to the following minimization problem:
$$\theta_{t+1} = \mathop{\text{argmin}}_\theta \nabla_\theta J(\theta_t)^T (\theta - \theta_t)\qquad \text{s.t.}\qquad \lVert\theta - \theta_t\rVert_2 \leq \varepsilon = \alpha\lVert\nabla_\theta J(\theta)\rVert$$
where $J(\theta_t) (\theta - \theta_t)$ is a first-order Taylor approximation for $J(\theta)$, ignoring the irrelevant constant term.

We can also understand learning policy parameters via the [[PG|policy gradient]] in this way. However, when $\theta$ parameterizes a policy, the constraint $\lVert\theta - \theta_t\rVert_2 \leq \varepsilon$ can have very different meanings depending on parameterization. This means artifacts of our policy parameterization can affect the policy gradient, which is not ideal.

To create a policy iteration method that is robust to policy parameterizaiton, we could try replacing the constraint $\lVert\theta - \theta_t\rVert_2 \leq \varepsilon$ with a different constraint that depends on $\pi_\theta$ and $\pi_{\theta_t}$ rather than $\theta$ and $\theta_t$ directly. 

## Strategy
Since $\pi_\theta(s)$ and $\pi_{\theta_t}(s)$ are distributions, we can quantify distance between them using the KL Divergence:
$$D_{\text{KL}}(\pi_{\theta_t}(\cdot\mid s)\vert\vert\pi_\theta(\cdot\mid s)) = \mathop{\mathbb{E}}_{a\sim\pi_{\theta_t}(s)}\log\frac{\pi_{\theta_t}(a\mid s)}{\pi_\theta(a\mid s)}$$
We can invent an overall similarity measure $D(\pi_{\theta_t}\vert\vert\pi_\theta)$ by taking an expectation with respect to $\mu^{\pi_{\theta_t}}(s\mid s_0)$, the state visitation distribution of the previous policy (which we can approximate later since we have samples from it):
$$\begin{align*}D(\pi_{\theta_t}\vert\vert\pi_\theta) &= \mathop{\mathbb{E}}_{s\sim\mu^{\pi_{\theta_t}}(s\mid s_0)}D_{\text{KL}}(\pi_{\theta_t}(\cdot\mid s)\vert\vert\pi_\theta(\cdot\mid s)) \\&= \mathop{\mathbb{E}}_{s\sim\mu^{\pi_{\theta_t}}(s\mid s_0)}\mathop{\mathbb{E}}_{a\sim\pi_{\theta_t}(s)}\log\frac{\pi_{\theta_t}(a\mid s)}{\pi_\theta(a\mid s)}\end{align*}$$
We can approximate the KL Divergence using a second-order Taylor approximation:
$$D_{\text{KL}}(\pi_{\theta_t}(\cdot\mid s)\vert\vert\pi_\theta(\cdot\mid s))\approx\frac{1}{2} (\theta -\theta_t)^TH(\theta-\theta_t)$$
where $H$ is the Hessian, given by
$$H = \mathop{\mathbb{E}}_{s\sim\mu^{\pi_{\theta_t}}(s\mid s_0)}\mathop{\mathbb{E}}_{a\sim\pi_{\theta_t}(s)}(\nabla_\theta \log \pi_\theta(a\mid s))(\nabla_\theta \log \pi_\theta(a\mid s))^T$$
which we can also approximate using trajectories from $\pi_{\theta_t}$.

The constrained optimization becomes
$$\theta_{t+1} = \mathop{\text{argmin}}_\theta \nabla_\theta J(\theta_t)^T (\theta - \theta_t)\qquad \text{s.t.}\qquad \frac{1}{2} (\theta -\theta_t)^TH(\theta-\theta_t) \leq \varepsilon$$
This is solved, e.g. via Lagrangian Duality, and we get
$$\theta_{t+1} \leftarrow \theta_t -  \sqrt{\frac{2\varepsilon}{\nabla_\theta J(\theta)^TH^{-1}\nabla_\theta J(\theta)}}H^{-1}\nabla_\theta J(\theta)$$
Storing $H$ and calculating $H^{-1}$ is difficult for highly parametric models, so we use the [[Conjugate Gradient Method]] to calculate $H^{-1}\nabla_\theta J(\theta)$. This only involves being able to evaluate $Hx$ for arbitrary vectors $x$, which we can do using the following formula derived below: 
$$Hx=\mathop{\mathbb{E}}_{s\sim\mu^{\pi_{\theta_t}}(s\mid s_0)}\mathop{\mathbb{E}}_{a\sim\pi_{\theta_t}(s)}\nabla_\theta\left[\left(-\nabla_\theta\log\pi_\theta(a\mid s)\right)^Tx\right]$$
## Analysis
#### Solving the Constrained Optimization
Here we show the solution to the following constrained optimization objective, where we assume the Hessian $H$ is positive-definite:
$$\begin{align*}\theta^*=&\mathop{\text{argmin}}_\theta \max_{\alpha\geq 0}  \left[\nabla_\theta J(\theta_t)^T (\theta - \theta_t) + \alpha\left(\frac{1}{2}(\theta -\theta_t)^TH(\theta-\theta_t) - \varepsilon\right)\right]\\=&\mathop{\text{argmin}}_\theta \max_{\alpha\geq 0}  \left[\nabla_\theta J(\theta_t)^T H^{-1}H(\theta - \theta_t) + \alpha\left(\frac{1}{2}[H(\theta -\theta_t)]^TH^{-1}[H(\theta-\theta_t)] - \varepsilon\right)\right]\end{align*}$$
Since the Hession is positive-definite and hence full rank, we can express this optimization in the variable $\phi=H(\theta-\theta_t)$. Defining $g = \nabla_\theta J(\theta)\vert_{\theta=\theta_t}$, we have
$$\theta^* = \theta_t +H^{-1}\mathop{\text{argmin}}_w \max_{\alpha\geq 0}  \left[g^T H^{-1}w + \alpha\left(\frac{1}{2}w^TH^{-1}w - \varepsilon\right)\right]$$
Since $g^T H^{-1}w$ is a scalar and $H$ is symmetric, we have
$$\begin{align*}(g^T H^{-1}w)^T &= g^T H^{-1}w\\\implies w^TH^{-1}g &= g^T H^{-1}w\end{align*}$$
Thus, we can write the optimization problem as
$$\begin{align*}\theta^*=&\theta_t + H^{-1}\mathop{\text{argmin}}_w \max_{\alpha\geq 0}  \frac{1}{2}\left[g^T H^{-1}w+w^T H^{-1}g + \alpha w^TH^{-1}w - \alpha \varepsilon\right]\\=&\theta_t + H^{-1}\mathop{\text{argmin}}_w \max_{\alpha\geq 0}  \frac{1}{2\alpha}\left[(g + \alpha w)^T H^{-1}(g +\alpha w)-g^TH^{-1}g - \alpha^2 \varepsilon\right]\end{align*}$$
At this point, we define the dual optimization objective
$$\begin{align*}&\max_{\alpha\geq 0}\mathop{\text{min}}_w  \frac{1}{2\alpha}\Big[(g + \alpha w)^T H^{-1}(g +\alpha w)-g^TH^{-1}g - \alpha^2 \varepsilon\Big]\\=&\max_{\alpha\geq 0} g(\alpha)\end{align*}$$
Note, however, that we can calculate $g(\alpha)$ specifically since only the first term depends on $w$:
$$\begin{align*}g(\alpha) &= \mathop{\text{min}}_w  \frac{1}{2\alpha}\Big[(g + \alpha w)^T H^{-1}(g +\alpha w)-g^TH^{-1}g - \alpha^2 \varepsilon\Big]\\&=\frac{1}{2\alpha}\left[-g^TH^{-1}g - \alpha^2 \varepsilon+\min_{w}(g + \alpha w)^T H^{-1}(g +\alpha w)\right]\end{align*}$$
Since $H$ is positive-definite, so is $H^{-1}$, so the minimization term is nonnegative. However, we can make this term 0 by choosing $w^*=-\frac{1}{\alpha}g$, so 0 must be the minimum, and
$$g(\alpha) = -\frac{g^TH^{-1}g}{2\alpha} - \alpha\varepsilon$$
Recall that the dual optimization objective was given by
$$\begin{align*}\max_{\alpha\geq 0} g(\alpha)&=\max_{\alpha\geq 0}\left[-\frac{g^TH^{-1}g}{2\alpha} - \alpha\varepsilon\right]\end{align*}$$
Notice that $$\lim_{\alpha\to\infty}g(\alpha)\to-\infty\qquad\text{ and }\qquad\lim_{\alpha\to 0^+}g(\alpha)\to-\infty$$ so this objective can only be maximized where $g'(\alpha) = 0$:
$$g'(\alpha) = \frac{g^TH^{-1}g}{2\alpha^2}-\varepsilon = 0$$
$$\implies\alpha^* = \sqrt{\frac{g^TH^{-1}g}{2\varepsilon}}$$
Since we've assumed $H$ is positive definite, this optimization problem is convex, so the dual solution $(\alpha^*, w^*)$ is also the solution to the primal (original) optimization problem. Thus
$$\begin{align*}\theta^* &= \theta_t + H^{-1}w^* \\&= \theta_t + \frac{1}{\alpha^*}H^{-1}g \\&= \theta_t + \sqrt{\frac{2\varepsilon}{g^TH^{-1}g}}H^{-1}g\end{align*}$$
which gives rise to the gradient update
$$\theta_{t+1} \leftarrow \theta_t -  \sqrt{\frac{2\varepsilon}{\nabla_\theta J(\theta)^TH^{-1}\nabla_\theta J(\theta)}}H^{-1}\nabla_\theta J(\theta)$$
#### Evaluating $Hx$ for arbitrary $x$
We solved the constrained optimization to find 
$$\theta_{t+1} \leftarrow \theta_t - \varepsilon \frac{H^{-1}\nabla_\theta J(\theta)}{\sqrt{\frac{1}{2}\nabla_\theta J(\theta)^TH^{-1}\nabla_\theta J(\theta)}}$$
If we can calculate the matrix-vector product $Hx$ for arbitrary $x$, we can use the conjugate gradient method to approximate $H^{-1}\nabla_\theta J(\theta)$, allowing us to calculate this optimization update.

Here, we use index notation and write
$$(Hx)_i = \sum_j \frac{\partial^2 D(\pi_{\theta_t}\vert\vert\pi_\theta)}{\partial \theta_i \partial \theta_j}x_j = \frac{\partial}{\partial\theta_i}\sum_j\frac{\partial D(\pi_{\theta_t}\vert\vert\pi_\theta)}{\partial \theta_j}x_j$$
This gives us the equation
$$\begin{align*}Hx &= \nabla_\theta\left[(\nabla_\theta D(\pi_{\theta_t}\vert\vert\pi_\theta))^Tx\right]\\&=\nabla_\theta\left[\left(-\mathop{\mathbb{E}}_{s\sim\mu^{\pi_{\theta_t}}(s\mid s_0)}\mathop{\mathbb{E}}_{a\sim\pi_{\theta_t}(s)}\nabla_\theta\log\pi_\theta(a\mid s)\right)^Tx\right]\\&=\mathop{\mathbb{E}}_{s\sim\mu^{\pi_{\theta_t}}(s\mid s_0)}\mathop{\mathbb{E}}_{a\sim\pi_{\theta_t}(s)}\nabla_\theta\left[\left(-\nabla_\theta\log\pi_\theta(a\mid s)\right)^Tx\right]\end{align*}$$

