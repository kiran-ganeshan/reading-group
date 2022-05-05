# Distribution Modeling with NNs

## Introduction
Suppose there's a distribution over a variable $y\in Y$ we want to know. By "want to know", we mean we want to be able to generate "good" samples $y'$. Suppose for generality that this distribution also depends on some other variable $x \in X$. We'll call the distribution $f(y\mid x)$. 

Suppose all we have are sample pairs $(x, y)$ where $y\sim f(y\mid x)$.  How can we build a flexible model to fit the sample pairs and learn $f$?

## Strategy

This occurs in four steps:
1. Decide on a differentiable **parameterization** $\xi: \mathbb{R}^p \to \mathcal{D}(A)$ to map numbers called **parameters** to distributions (which often make up a larger family).
2. Learn $f_\theta: X\to \mathbb{R}^p$ to map context info to distribution parameters
4. Design a loss function $L(x, y, \theta)$ to tell when parameters $\theta$ produce "good" samples. When "good" means that samples match the dataset, we can use the negative log-likelihood: $-\log\xi(y\mid f_\theta(x))$. (Other losses may be appropriate for different definitions of good e.g. achieves good reward in RL).
5. Design a mechanism to sample from $\xi(a\mid\phi)$ for any $\phi\in\mathbb{R}^p$.

Having accomplished these steps, we can approximate $f(y\mid x) \approx \xi(y\mid f_\theta(x))$ and simply perform backpropagation on $L(x, y, \theta)$ with respect to  $\theta$ in order to optimize our predicted distribution.

## Examples

- If $Y$ is discrete and finite:
	1. Represent distributions over the action space as finite lists of probabilites of length $\vert Y\vert$ (this means our NN will output $p = \vert Y\vert$ values). We must define our parameterization so that for any neural network output $\phi\in\mathbb{R}^{\vert Y\vert}$, the probabilities $\xi(y\mid \phi)$ are positive and add to 1. The softmax function satisfies this requirement:
	$$\xi(y\mid \phi) = \mathop{\text{softmax}}(\phi) =\frac{\exp\Big(\frac{1}{\alpha}\phi_y\Big)}{\sum_{y'\in A}\exp\Big(\frac{1}{\alpha}\phi_{y'}\Big)}$$
	where $\phi_y$ is the component of the neural network output $\phi$ corresponding to $y$.
	3. Run inference to obtain $\phi$. Sample $a$ from the categorical random variable $\xi(y\mid \phi)$, e.g. via rejection sampling or by dividing the unit interval into $\vert Y\vert$ appropriately sized parts.
	
	This way, our neural network could be trained to output any categorical distribution over the action space.
- If $Y$ is infinite, whether continuous or discrete, we cannot map the space of distributions over $Y$ to $\mathbb{R}^n$ for any $n$. Thus, we must choose a useful family of distributions to parametrize with the NN. When $Y$ is continuous, one common choice is the family of Gaussians:
	1. Represent distributions over $A = \mathbb{R}^n$ using a Gaussian with mean $\mu \in \mathbb{R}^n$ and covariance matrix $\Sigma \in \mathbb{R}^{n^2}$. Since $\Sigma$ only contains $\frac{n(n-1)}{2}$ free parameters, our whole parameterization requires $p = \frac{1}{2}n(n+1)$. Given a neural net output $\phi = (\mu, \Sigma)$, we have
	$$\xi(a\mid \mu, \Sigma) = \frac{1}{\sqrt{2\pi\vert \Sigma\vert}}\exp\bigg(-\frac{1}{2} (a - \mu)^T \Sigma^{-1} (a - \mu)^T\bigg)$$
	3. Run inference to obtain $(\mu, \Sigma) = f_\theta(s)$. Then we sample $\varepsilon \sim \mathcal{N}(\vec 0, I_n)$ and output $\mu + \Sigma\varepsilon$ as our sample, where we take a matrix-vector product between $\Sigma$ and $\varepsilon$.
- A more complex but highly versatile family is the Mixture of Gaussians (MoG):
	1. Represent distributions over $A = \mathbb{R}^n$ using a mixture of $m$ Gaussians with mean $\mu_i \in \mathbb{R}^n$, covariance matrix $\Sigma_i \in \mathbb{R}^{n^2}$, and mixture weights $w_i \in \mathbb{R}$. Our whole parameterization requires $p = \frac{1}{2}mn(n+1)$ many parameters. Given a neural net output $\phi = (\mu_i, \Sigma_i, w_i)_{i\in[m]}$, we have
	$$\xi(a\mid \phi) = \sum_{i=1}^m\frac{w_i}{\sqrt{2\pi\vert \Sigma_i\vert}}\exp\bigg(-\frac{1}{2} (a - \mu_i)^T \Sigma_i^{-1} (a - \mu_i)^T\bigg)$$
	3. To sample from this distribution, run inference to obtain the distribution parameters $(\mu_i, \Sigma_i, w_i)$. Sample $j\in[m]$ according to the probabilites given by $w_i$, and then sample from the Gaussian with parameters $(\mu_j, \Sigma_j)$.