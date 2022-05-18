# Double Q-Learning
Requirements: [[DQN]]
## Resources
- [Towards Data Science](https://towardsdatascience.com/double-deep-q-networks-905dd8325412)
- [van Hasselt et al. (2010)](https://arxiv.org/pdf/1509.06461.pdf), which introduced double Q learning
- [van Hasselt et al. (2015)](https://arxiv.org/pdf/1509.06461.pdf), which refined double Q learning in light of new developments after the release of DQN

## Introduction
Double Q-learning refers to two techniques invented to combat overestimation in Q-learning. The original Double Q-Learning technique came before DQN, when Q-learning did not use neural networks. When DQN was released, a simplified version of the Double Q-learning technique for use with neural networks was invented. Both have influence in modern RL, so we will present both. 

Both technqiues make use of the following observation:

In the standard Q-learning setup, we gather transitions $(s, a, r, s')$ and train a neural network $Q_\theta(s, a)$ to satisfy
$$\begin{align*}Q_\theta(s, a) &= r + \gamma \max_{a'\in A}Q_{\theta'}(s', a')\\&=r + \gamma Q_{\theta'}\left(s', \mathop{\text{argmax}}_{a'\in A}Q_{\theta'}(s', a')\right)\end{align*}$$
where $\theta'$ are the parameters of the target network. As we show above, the maximum over the target network $Q_{\theta'}$ subtly involves an argmax over $Q_{\theta'}$ and a subsequent evaluation of $Q_{\theta'}$.

 This raises an interesting question: should we be using the same network to find the optimal next action as we use to evaluate this action?

## Strategy

#### Original Double Q-Learning
The first implementation of Double Q Learning introduced two networks, $Q^A$ and $Q^B$, with parameters $\theta$ and $\phi$ respectively.  These networks are learned to satisfy
$$\begin{align*}Q^A_\theta(s, a) &= r +  \gamma Q^B_\phi\left(s', \mathop{\text{argmax}}_{a'\in A}Q^A_\theta(s', a')\right)\\Q^B_\phi(s, a)&=r + \gamma Q^A_\theta\left(s', \mathop{\text{argmax}}_{a'\in A}Q^B_\phi(s', a')\right)\end{align*}$$
That is, we always use a network trained on identical data with different parameters to calculate Bellman targets, while using the same network we are training to select actions.

#### Modern Double Q-Learning

^96650b

While the above technique uses two indepedently initialized neural networks to achieve stability in the Bellman recurrence, [[DQN]] achieves this simply by evaluating the maximum over a target network $Q_{\theta'}$, whose parameters $\theta'$ track the trained parameters $\theta$ via Polyak averaging:
$$\theta' \leftarrow \alpha \theta + (1 - \alpha) \theta'$$
It turns out we can replace the extra network $Q^B_\phi$ with a target network to make Double Q-Learning more efficient. We simply train $Q_\theta$ to satisfy the following:
$$Q_\theta(s, a) = r + \gamma Q_{\theta'}\left(s', \mathop{\text{argmax}}_{a'\in A}Q_\theta(s', a')\right)$$

 
 