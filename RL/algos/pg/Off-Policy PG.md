# Off-Policy PG
Requirements: [[PG]], [[RL Paradigms]]

## Introduction
We have seen that the policy gradient is the epitome of an **on-policy** algorithm, meaning we must train policies on data produced by the policy itself. Specifically, the states we train on must be distributed according to the policy's state vistiation distribution $\mu^\pi(s\mid s_0)$, and the actions must be distributed according to the policy itself $\pi(a\mid s)$.

However, when we update our policy using the policy gradient, it will change, meaning state-aciton pairs from an old trajectory will be distributed according to $s\sim \mu^{\pi_{\phi}}(s\mid s_0)$ and $a\sim\pi_\phi(s)$ for some old policy parameters $\phi$. 

For this reason, traditional policy gradient algorithms train only on data gathered since the previous update, which impedes our ability to train on past information. Can we adjust the policy gradient theorem to learn from older data?

## Strategy
We can change the distributions with respect to which we are taking expectations using [[Importance Sampling]] if we know the distributions in question. 

In this case, if $\theta$ are the new policy parameters and $\phi$ are the old policy parameters, we can change both
1. the policy sampled from in the inner expectation from $\pi_\theta$ to $\pi_\phi$ by multiplying the expression in the expectation by the importance weight $$\frac{\pi_\theta(a\mid s)}{\pi_\phi(a\mid s)}$$
3. the state visitation distribution sampled from in the outer expectation from $\mu^{\pi_\theta}$ to $\mu^{\pi_\phi}$ by multiplying the expression in the expectation by the importance weight$$\frac{\mu^{\pi_\theta}(s_t\mid s_0)}{\mu^{\pi_\phi}(s_t\mid s_0)}$$

This results in an Offline Policy Gradient:
$$\begin{align*}\nabla_{\theta}J(\theta) &\propto \mathop{\mathbb{E}}_{s\sim\mu^{\pi_\theta}(s\mid s_0)}\; 
\mathop{\mathbb{E}}_{a\sim\pi_\theta(s)}\Big[Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\Big]\\&\propto \mathop{\mathbb{E}}_{s\sim\mu^{\pi_\phi}(s\mid s_0)}
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\frac{\mu^{\pi_\theta}(s\mid s_0)\pi_\theta(a\mid s)}{\mu^{\pi_\phi}(s\mid s_0)\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\end{align*}$$
Using a very rough Monte-Carlo approximation, we can derive an estimator for the importance weights:
$$\frac{\mu^{\pi_\theta}(s_t\mid s_0)\pi_\theta(a_t\mid s_t)}{\mu^{\pi_\phi}(s_t\mid s_0)\pi_\phi(a_t\mid s_t)}\approx \prod_{t'=0}^{t}\frac{\pi_\theta(a_{t'}\mid s_{t'})}{\pi_\phi(a_{t'}\mid s_{t'})}$$
This yields a Monte-Carlo approximation of the off-policy policy gradient:
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
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\frac{\pi_\theta(a\mid s)}{\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\right\}\\&\propto \sum_{s\in S}\left\{\mu^{\pi_\phi}(s\mid s_0)
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\frac{\mu^{\pi_\theta}(s\mid s_0)\pi_\theta(a\mid s)}{\mu^{\pi_\phi}(s\mid s_0)\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\right\}\\&\propto \mathop{\mathbb{E}}_{s\sim\mu^{\pi_\phi}(s\mid s_0)}
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\frac{\mu^{\pi_\theta}(s\mid s_0)\pi_\theta(a\mid s)}{\mu^{\pi_\phi}(s\mid s_0)\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\end{align*}$$
The simplest way to approximate
$$\frac{\mu^{\pi_\theta}(s\mid s_0)\pi_\theta(a\mid s)}{\mu^{\pi_\phi}(s\mid s_0)\pi_\phi(a\mid s)}$$
actually starts by separating the policy gradient into timesteps. We define $\mu_t^{\pi}(s\mid s_0)$ as the distribution of states $s_t$ at timestep $t$ under poilcy $\pi$. Then
$$\begin{align*}\nabla_{\theta}J(\theta) &\propto \mathop{\mathbb{E}}_{s\sim\mu^{\pi_\phi}(s\mid s_0)}
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\frac{\mu^{\pi_\theta}(s\mid s_0)\pi_\theta(a\mid s)}{\mu^{\pi_\phi}(s\mid s_0)\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\\&\propto \frac{1}{T}\sum_{t=0}^{T-1} \mathop{\mathbb{E}}_{s\sim\mu_t^{\pi_\phi}(s\mid s_0)}
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\frac{\mu_t^{\pi_\theta}(s\mid s_0)\pi_\theta(a\mid s)}{\mu_t^{\pi_\phi}(s\mid s_0)\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\end{align*}$$
We can then derive the following Monte-Carlo approximation for the state distribution at time $t$:
$$\begin{align*}\mu_t^{\pi_\theta}(s_t\mid s_0)&=\sum_{s_{t'}, a_{t'}} \prod_{t'=0}^{t-1}\pi_\theta(a_{t'}\mid s_{t'})\mathcal{T}(s_{t'+1}\mid s_{t'}, a_{t'})\\&\approx \prod_{t'=0}^{t-1}\pi_\theta(a_{t'}\mid s_{t'})\mathcal{T}(s_{t'+1}\mid s_{t'}, a_{t'})\end{align*}$$
which yields
$$\begin{align*}\frac{\mu_t^{\pi_\theta}(s_t\mid s_0)\pi_\theta(a_t\mid s_t)}{\mu_t^{\pi_\phi}(s_t\mid s_0)\pi_\phi(a_t\mid s_t)}&\approx \frac{\pi_\theta(a_t\mid s_t)}{\pi_\phi(a_t\mid s_t)}\prod_{t'=0}^{t-1}\frac{\pi_\theta(a_{t'}\mid s_{t'})\mathcal{T}(s_{t'+1}\mid s_{t'}, a_{t'})}{\pi_\phi(a_{t'}\mid s_{t'})\mathcal{T}(s_{t'+1}\mid s_{t'}, a_{t'})}\\&\approx \prod_{t'=0}^{t}\frac{\pi_\theta(a_{t'}\mid s_{t'})}{\pi_\phi(a_{t'}\mid s_{t'})}\end{align*}$$
Using this (rather unjustified, but best available) Monte-Carlo approximation in the off-policy policy gradient, we have
$$\begin{align*}\nabla_{\theta}J(\theta) &\propto \frac{1}{T}\sum_{t=0}^{T-1} \mathop{\mathbb{E}}_{s\sim\mu_t^{\pi_\phi}(s\mid s_0)}
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\frac{\mu_t^{\pi_\theta}(s\mid s_0)\pi_\theta(a\mid s)}{\mu_t^{\pi_\phi}(s\mid s_0)\pi_\phi(a\mid s)}Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\\&\propto \frac{1}{T}\sum_{t=0}^{T-1} \mathop{\mathbb{E}}_{s\sim\mu_t^{\pi_\phi}(s\mid s_0)}
\mathop{\mathbb{E}}_{a\sim\pi_\phi(s)}\left[\left(\prod_{t'=0}^{t}\frac{\pi_\theta(a_{t'}\mid s_{t'})}{\pi_\phi(a_{t'}\mid s_{t'})}\right)Q^{\pi_\theta}(s, a)\nabla_{\theta} \ln\pi_\theta(a\mid s)\right]\end{align*}$$
Approximating $\mu_t^{\pi_\phi}$ and $\pi_\phi$ using the states and actions from step $t$ of each rollout, and using the trajectory total reward (which is technically the Monte-Carlo approximation for $Q^{\pi_\phi}$) as an estimator for $Q^{\pi_\theta}$, we derive the (rather imperfect) estimator:
$$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^N\sum_{t=0}^T\Bigg(\prod_{t'=0}^{t}\frac{\pi_\theta(a_{t'}\mid s_{t'})}{\pi_\phi(a_{t'}\mid s_{t'})}\Bigg)\Bigg(\sum_{t'=t}^T \gamma^{t' - t} r^{(i)}_{t'}\Bigg) \nabla_\theta\ln\pi_\theta\Big(a_t^{(i)}\mid s_t^{(i)}\Big)$$
This can be very well improved on by methods that attempt to bound the difference between state visitation distributions, meaning we can ignore their ratio rather than calculating it with Monte-Carlo approximation.