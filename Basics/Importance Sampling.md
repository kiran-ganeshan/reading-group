# Importance Sampling
## Introduction
Suppose we have samples $\{x_i\}_{i=1}^n$ from a distribution $\mathcal{D}$ over a space $\mathcal{X}$, and we would like to estimate $\E_{x\sim\mathcal{D}} f(x)$ for some function $f$. We can use the following estimator:
$$\E_{x\sim\mathcal{D}} f(x)\approx \frac{1}{n}\sum_{i=1}^nf(x_i)$$
But what if we have samples $\{x_i'\}_{i=1}^n$ from a different distribution $\mathcal{D'}$? Can we still build an estimator for the expected value  of an arbitrary function $f$ under the original distribution $\mathcal{D}$?

It turns out this is possible if we can calculate the probabilities of our new samples under both distributions ($\mathcal{D}(x_i')$ and $\mathcal{D'}(x_i')$) . Note that
$$\begin{align*}\E_{x\sim\mathcal{D}}f(x) &= \sum_{x\in \mathcal{X}}\mathcal{D}(x)f(x) \\&= \sum_{x\in \mathcal{X}}\mathcal{D'}(x)\frac{\mathcal{D}(x)}{\mathcal{D'}(x)}f(x) \\&= \E_{x\sim \mathcal{D'}} \left[\frac{\mathcal{D}(x)}{\mathcal{D'}(x)}f(x)\right]\end{align*}$$
So we can define $g(x) = \frac{\mathcal{D}(x)}{\mathcal{D'}(x)}f(x)$, and then by the same logic as above we can derive an estimator:
$$\E_{x\sim\mathcal{D}}f(x) = \E_{x\sim\mathcal{D'}}g(x) \approx \frac{1}{n}\sum_{i=1}^n g(x_i') = \frac{1}{n}\sum_{i=1}^n \frac{\mathcal{D}(x_i')}{\mathcal{D'}(x_i')}f(x_i')$$

## Definition
Given a pair of distribution $\mathcal{D}$ and $\mathcal{D'}$ over a space $\mathcal{X}$, and samples $\{x_i'\}_{i=1}^n$ where $x_i'\sim \mathcal{D'}$, we can approximate an expectation $\E_{x\sim\mathcal{D}}f(x)$ for any $f:\mathcal{X}\to\R$ using the **importance sampled estimator**
$$\E_{x\sim\mathcal{D}}f(x)\approx  \frac{1}{n}\sum_{i=1}^n \frac{\mathcal{D}(x_i')}{\mathcal{D'}(x_i')}f(x_i')$$
where we have multiplied $f$ by the **importance weight** $\frac{\mathcal{D}(x)}{\mathcal{D'}(x)}$, which accounts for the differences between the distributions $\mathcal{D}$ and $\mathcal{D'}$.
