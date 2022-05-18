# Distribution Modeling with NNs

## Introduction
Suppose there's a distribution over a variable $y\in Y$ we want to know. By "want to know", we mean we either
- want to be able to generate samples $y'$ that look like they came from the distribution, or
- want to be able to learn properties, e.g. the mean, median, or $k$th percentile, of some function $g(y)$ with respect to the distribution

Suppose for generality that this distribution also depends on some other variable $x \in X$. We'll call the distribution $f(y\mid x)$. 

Suppose all we have are sample pairs $(x, y)$ where $y\sim f(y\mid x)$.  We will design separate strategies to accomplish each of these goals in the setting described.

## Sampling via Parameterization
Suppose we want to build a model of $f(y\mid x)$ that allows us to sample. This occurs in four steps:
1. Decide on a differentiable **parameterization** $\xi: \mathbb{R}^p \to \mathcal{D}(Y)$ to map numbers called **parameters** to distributions (which often make up a larger family).
2. Learn $h_\theta: X\to \mathbb{R}^p$ to map context info to distribution parameters
4. Design a loss function $L(x, y, \theta)$ to tell when parameters $\theta$ produce "good" samples. Most often, "good" means that samples match the dataset, and we can use the negative log-likelihood: $-\log\xi(y\mid h_\theta(x))$. Other losses may be appropriate for different definitions of "good," e.g. achieves good reward in RL.
5. Design a mechanism to sample from $\xi(a\mid\phi)$ for any $\phi\in\mathbb{R}^p$.

Having accomplished these steps, we can approximate $f(y\mid x) \approx \xi(y\mid h_\theta(x))$ and simply perform backpropagation on $L(x, y, \theta)$ with respect to  $\theta$ in order to optimize our predicted distribution.

When we want to sample $y$ given conditional info $x$, simply  use the mechanism from part (4) to sample from $\xi(y\mid h_\theta(x))$.

#### Differentiation with respect to Samples

What if the loss function we want to minimize in (3) depends not on the probability of a sample, but on the sample itself? Here we can utilize something called the **reparameterization trick**. The basic idea is to express a sample $y\sim\xi(y\mid \phi)$ in terms of a sample from a "standard" distribution with constant parameters $\phi_0$:
$$y\sim\xi(y\mid\phi) \quad\leftrightarrow \quad y= f_\phi(y'), \;y'\sim\xi(y\mid\phi_0)$$
Then, provided $f_\phi$ is differentiable with respect to $\phi$, we can simply sample $y'\sim \xi(y\mid\phi_0)$, treat $y'$ as constant, and then calculate $\nabla_\phi y = \nabla_\phi f_\phi(y')$.

The most common example is the Gaussian case, where $\phi = h_\theta(x) = (\mu_\theta(x), \Sigma_\theta(x))$:
$$y = f_\phi(y') = \mu_\theta(x) + \Sigma_\theta(x) y', \quad y'\sim\mathcal{N}(0, I)$$
$$\nabla_\theta y  = \nabla_\theta\mu_\theta(x) + (\nabla_\theta\Sigma_\theta(x))y', \quad y'\sim\mathcal{N}(0, I)$$


## Quantile/Expectile Regression
We can learn quantiles with **quantile regression**, and we can learn expectiles with **expectile regression**. (You can think of expectiles like quantiles for the mean: just like the 0.5 quantile is the median, the 0.5 expectile is the mean.)

- In **quantile regression**, we learn a function $h_\theta(x)$ whose output is the $\tau$-quantile of $g(y)$ where $y\sim f(y\mid x)$. The following loss accomplishes this:
$$\rho_\tau(x, y, \theta) = \tau \max(g(y) - h_\theta(x), 0) + (1 - \tau)\max(h_\theta(x) - g(y), 0)$$
notice when $\tau=0.5$, we are attempting to learn the median, and $\rho_\tau$ becomes the L1 loss of the residual:
$$\rho_{0.5}(x, y, \theta) = \frac{1}{2}\lvert h_\theta(x) - g(y)\rvert$$
- In **expectile regression**, we learn a function $h_\theta(x)$ whose output is the $\tau$-expectile of $g(y)$ where $y\sim f(y\mid x)$. The following loss accomplishes this:
$$\rho^{(2)}_\tau(x, y, \theta) = \tau \max(g(y) - h_\theta(x), 0)^2 + (1 - \tau)\max(h_\theta(x) - g(y), 0)^2$$
notice when $\tau=0.5$, we are attempting to learn the mean, and $\rho_\tau$ becomes the L2 loss of the residual:
$$\rho^{(2)}_{0.5}(x, y, \theta) = \frac{1}{2}(h_\theta(x) - g(y))^2$$


## Examples
- Sample learning when $Y$ is discrete and finite:
	1. Represent distributions over the action space as finite lists of probabilites of length $\vert Y\vert$ (this means our NN will output $p = \vert Y\vert$ values). We must define our parameterization so that for any neural network output $\phi\in\mathbb{R}^{\vert Y\vert}$, the probabilities $\xi(y\mid \phi)$ are positive and add to 1. The softmax function satisfies this requirement:
	$$\xi(y\mid \phi) = \mathop{\text{softmax}}(\phi) =\frac{\exp\Big(\frac{1}{\alpha}\phi_y\Big)}{\sum_{y'\in A}\exp\Big(\frac{1}{\alpha}\phi_{y'}\Big)}$$
	where $\phi_y$ is the component of the neural network output $\phi$ corresponding to $y$.
	3. Run inference to obtain $\phi$. Sample $a$ from the categorical random variable $\xi(y\mid \phi)$, e.g. via rejection sampling or by dividing the unit interval into $\vert Y\vert$ appropriately sized parts.
	
	This way, our neural network could be trained to output any categorical distribution over the action space.
- Sample learning when $Y$ is continuous using the Gaussian family of distributions: 
	1. Represent distributions over $Y = \mathbb{R}^n$ using a Gaussian with mean $\mu \in \mathbb{R}^n$ and covariance matrix $\Sigma \in \mathbb{R}^{n^2}$. Since $\Sigma$ only contains $\frac{n(n-1)}{2}$ free parameters, our whole parameterization requires $p = \frac{1}{2}n(n+1)$. Given a neural net output $\phi = (\mu, \Sigma)$, we have
	$$\xi(y\mid \mu, \Sigma) = \frac{1}{\sqrt{2\pi\vert \Sigma\vert}}\exp\bigg(-\frac{1}{2} (y - \mu)^T \Sigma^{-1} (y - \mu)^T\bigg)$$
	3. Run inference to obtain $(\mu, \Sigma) = f_\theta(s)$. Then we sample $\varepsilon \sim \mathcal{N}(\vec 0, I_n)$ and output $\mu + \Sigma\varepsilon$ as our sample, where we take a matrix-vector product between $\Sigma$ and $\varepsilon$.
- Sample learning when $Y$ is continuous using Mixtures of Gaussians (MoG), a more complex but highly versatile family:
	1. Represent distributions over $Y = \mathbb{R}^n$ using a mixture of $m$ Gaussians with mean $\mu_i \in \mathbb{R}^n$, covariance matrix $\Sigma_i \in \mathbb{R}^{n^2}$, and mixture weights $w_i \in \mathbb{R}$. Our whole parameterization requires $p = \frac{1}{2}mn(n+1)$ many parameters. Given a neural net output $\phi = (\mu_i, \Sigma_i, w_i)_{i\in[m]}$, we have
	$$\xi(y\mid \phi) = \sum_{i=1}^m\frac{w_i}{\sqrt{2\pi\vert \Sigma_i\vert}}\exp\bigg(-\frac{1}{2} (y - \mu_i)^T \Sigma_i^{-1} (y - \mu_i)^T\bigg)$$
	3. To sample from this distribution, run inference to obtain the distribution parameters $(\mu_i, \Sigma_i, w_i)$. Sample $j\in[m]$ according to the probabilites given by $w_i$, and then sample from the Gaussian with parameters $(\mu_j, \Sigma_j)$.
- Learning the conditional median of $y$: represent the conditional median using a neural network $h_\theta(x)$ and then optimize the loss
$$\rho_{0.5}(x, y, \theta) = \frac{1}{2}\lvert y - h_\theta(x)\rvert$$
- Learning the conditional mean of $y^3 + y$: represent the conditional mean using a neural network $h_\theta(x)$ and then optimize the loss
$$\rho^{(2)}_{0.5}(x, y, \theta) = \frac{1}{2}\lvert y^3 + y - h_\theta(x)\rvert^2$$
Notice this means that  ^258e8c
- Learning the conditional maximum of an arbitrary, tractable $g(y)$: represent the conditional max using a neural network $h_\theta(x)$ and then optimize either of the following losses:
$$\rho_{1}(x, y, \theta) =   \max(g(y) - h_\theta(x), 0) $$
$$\rho^{(2)}_{1}(x, y, \theta) = \max(g(y) - h_\theta(x), 0)^2$$
Note this will result in stability issues, since the network can learn to place $h_\theta(x)$ arbitrarily high and zero out the loss, despite not having learned the correct expectile. To avoid this, never use $\tau=1$ directly, simply using $\tau = 1-\varepsilon$ for small $\varepsilon$.
