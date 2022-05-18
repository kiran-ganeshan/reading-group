# Importance Sampling for PG
Requirements: [[PG]]

## Introduction
Recall that the policy gradient theorem tells us
$$\begin{align*}\nabla_{\theta}J(\theta) &\propto \mathop{\mathbb{E}}_{s\sim\mu^{\pi}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi(s)}\left[Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\end{align*}$$
which is great, since the state-action pairs $(s, a)$ visited by the policy $\pi_\theta$ are distributed according to $s\sim\mu^{\pi_\theta}(s\mid s_0)$ and $a\sim\pi_\theta(s)$. Hence we can approximate these expectations by sampling past state-action pairs.

However, when we update our policy, it will change, meaning state-aciton pairs from an old trajectory will be distributed according to $s\sim \mu^{\pi_{\phi}}(s\mid s_0)$ and $a\sim\pi_\phi(s)$ for some old policy parameters $\phi$. Can we adjust the policy gradient theorem to learn from this data?

## Strategy
We can change the distributions with respect to which we are taking expectations using [[Importance Sampling]] if we know the distributions in question. 

In this case, if $\theta$ are the new policy parameters and $\phi$ are the old policy parameters, we can change both
1. the policy sampled from in the inner expectation from $\pi_\theta$ to $\pi_\phi$ by multiplying the expression in the expectation by the importance weight $$\frac{\pi_\theta(a\mid s)}{\pi_\phi(a\mid s)}$$
3. the state visitation distribution sampled from in the outer expectation from $\mu^{\pi_\theta}$ to $\mu^{\pi_\phi}$ by multiplying the expression in the expectation by the importance weight$$\frac{\mu^{\pi_\theta}(s_t\mid s_0)}{\mu^{\pi_\phi}(s_t\mid s_0)}$$

This results in an Offline Policy Gradient:
$$\begin{align*}\nabla_{\theta}J(\theta) &\propto \mathop{\mathbb{E}}_{s\sim\mu^{\pi_\theta}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi_\theta(s)}\Big[Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\Big]\\&\propto \mathop{\mathbb{E}}_{s\sim\mu^{\pi_\phi}(s\mid s_0)}
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\frac{\mu^{\pi_\theta}(s\mid s_0)\pi_\theta(a\mid s)}{\mu^{\pi_\phi}(s\mid s_0)\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\end{align*}$$
The state visitation distribution is intractable, but we can approximate it by assuming that the current trajectory enumerates the only path to $s_t$:
$$\frac{\mu^{\pi_\theta}(s_t\mid s_0)\pi_\theta(a_t\mid s_t)}{\mu^{\pi_\phi}(s_t\mid s_0)\pi_\phi(a_t\mid s_t)}\approx \prod_{t'=0}^{t}\frac{\pi_\theta(a_{t'}\mid s_{t'})}{\pi_\phi(a_{t'}\mid s_{t'})}$$
This yields the final approximation of the policy gradient:
$$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^N\sum_{t=0}^T\Bigg(\prod_{t'=0}^{t}\frac{\pi_\theta(a_{t'}\mid s_{t'})}{\pi_\phi(a_{t'}\mid s_{t'})}\Bigg)\Bigg(\sum_{t'=t}^T \gamma^{t' - t} r^{(i)}_{t'}\Bigg) \nabla_\theta\ln\pi_\theta\Big(a_t^{(i)}\mid s_t^{(i)}\Big)$$
meaning that we can effectively optimize our policy using this gradient by maximizing the surrogate objective
$$\frac{1}{N}\sum_{i=1}^N\sum_{t=0}^T\Bigg(\prod_{t'=0}^{t}\frac{\operatorname{sg}(\pi_\theta(a_{t'}\mid s_{t'}))}{\pi_\phi(a_{t'}\mid s_{t'})}\Bigg)\Bigg(\sum_{t'=t}^T \gamma^{t' - t} r^{(i)}_{t'}\Bigg) \ln\pi_\theta\Big(a_t^{(i)}\mid s_t^{(i)}\Big)$$
where $\operatorname{sg}$ is a stop-gradient function meant to indicate that the expression it acts on is treated as constant for backpropagation.
## Analysis
The full derivation for the offline policy gradient is as follows:
$$\begin{align*}\nabla_{\theta}J(\theta) &\propto \mathop{\mathbb{E}}_{s\sim\mu^{\pi_\theta}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi_\theta(s)}\Big[Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\Big]\\&\propto \mathop{\mathbb{E}}_{s\sim\mu^{\pi_\theta}(s\mid s_0)}\; 
\sum_{a\in A}\Big[\pi_\theta(a\mid s)Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\Big]\\&\propto \mathop{\mathbb{E}}_{s\sim\mu^{\pi_\theta}(s\mid s_0)}\; 
\sum_{a\in A}\left[\pi_\phi(a\mid s)\frac{\pi_\theta(a\mid s)}{\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\\&\propto \mathop{\mathbb{E}}_{s\sim\mu^{\pi_\theta}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\frac{\pi_\theta(a\mid s)}{\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\\&\propto \sum_{s\in S}\left\{\mu^{\pi_\theta}(s\mid s_0) 
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\frac{\pi_\theta(a\mid s)}{\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\right\}\\&\propto \sum_{s\in S}\left\{\mu^{\pi_\phi}(s\mid s_0)\frac{\mu^{\pi_\theta}(s\mid s_0)}{\mu^{\pi_\phi}(s\mid s_0)}
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\frac{\pi_\theta(a\mid s)}{\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\right\}\\&\propto \sum_{s\in S}\left\{\mu^{\phi}(s\mid s_0)
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\frac{\mu^{\pi_\theta}(s\mid s_0)\pi_\theta(a\mid s)}{\mu^{\pi_\phi}(s\mid s_0)\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\right\}\\&\propto \mathop{\mathbb{E}}_{s\sim\mu^{\phi}(s\mid s_0)}
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\frac{\mu^{\pi_\theta}(s\mid s_0)\pi_\theta(a\mid s)}{\mu^{\pi_\phi}(s\mid s_0)\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\end{align*}$$
The simplest way to approximate
$$\frac{\mu^{\pi_\theta}(s\mid s_0)\pi_\theta(a\mid s)}{\mu^{\pi_\phi}(s\mid s_0)\pi_\phi(a\mid s)}$$
is using the following Monte-Carlo approximation for the state $s_t$ which appears at step $t$ in a trajectory:
$$\begin{align*}\mu^{\pi_\theta}(s_t\mid s_0)&=\sum_{s_{t'}, a_{t'}} \prod_{t'=0}^{t-1}\pi_\theta(a_{t'}\mid s_{t'})\mathcal{T}(s_{t'+1}\mid s_{t'}, a_{t'})\\&\approx \prod_{t'=0}^{t-1}\pi_\theta(a_{t'}\mid s_{t'})\mathcal{T}(s_{t'+1}\mid s_{t'}, a_{t'})\end{align*}$$
which yields
$$\begin{align*}\frac{\mu^{\pi_\theta}(s_t\mid s_0)\pi_\theta(a_t\mid s_t)}{\mu^{\pi_\phi}(s_t\mid s_0)\pi_\phi(a_t\mid s_t)}&\approx \frac{\pi_\theta(a_t\mid s_t)}{\pi_\phi(a_t\mid s_t)}\prod_{t'=0}^{t-1}\frac{\pi_\theta(a_{t'}\mid s_{t'})\mathcal{T}(s_{t'+1}\mid s_{t'}, a_{t'})}{\pi_\phi(a_{t'}\mid s_{t'})\mathcal{T}(s_{t'+1}\mid s_{t'}, a_{t'})}\\&\approx \prod_{t'=0}^{t}\frac{\pi_\theta(a_{t'}\mid s_{t'})}{\pi_\phi(a_{t'}\mid s_{t'})}\end{align*}$$
This approximation is quite imperfect, and
