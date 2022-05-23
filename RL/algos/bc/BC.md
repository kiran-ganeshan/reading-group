# Behavioral Cloning
## Introduction
If we'd like to use a neural network to solve a complex control task, one of the simplest ideas is to train a neural network to imitate an expert human performing the task. This is called **Imitation Learning** or **Behavioral Cloning** (BC). We can even deal with suboptimal performance in the dataset of human demonstrations by filtering out the bottom $x$% of trajectories in terms of total reward. This is often referred to as %BC.
## Strategy
Behavioral cloning requires gathering state-action pairs from expert demonstration $$\left\{\left(s_t^{(i)}, a_t^{(i)}\right)_{t=1}^T\right\}_{i=1}^N$$
where 
- $s_t^{(i)}$ is the state in step $t$ of the $i$th trajectory, 
- $a_t^{(i)}$ is the action in step $t$ of the $i$th trajectory, 
- $T$ is the max number of timesteps per demonstration, and 
- $N$ is the number of demonstration.
## Use Cases
Behavioral cloning works in
- discrete and continuous state spaces
- discrete and continuous action spaces

## Algorithm
**%BC**: Behavioral cloning

**Parameters**: 
- Percentage $x$ of dataset to filter out
- Number of total steps $T$
- Replay Buffer capacity $N$
- Number of iterations $K$
- Learning rate

**Algorithm**:
> Initialize expert dataset $\mathcal{D} = \left\{\left(s_t^{(i)}, a_t^{(i)}\right)_{t=1}^T\right\}_{i=1}^N$ \
> Remove low-reward trajectories from $\mathcal{D}$ (bottom $x$%) \
> Initialize policy NN $\pi_\theta:S\to A$ with random weights $\theta$\
> **for** $k\in\{1, \dots, K\}$ **do**:\
> $\qquad$ Set our loss to $L(\theta) = \sum_{i=1}^N\sum_{t=1}^T \left(a_t^{(i)} - \pi_\theta\left(s_t^{(i)}\right)\right)^2$\
> $\qquad$ Perform gradient descent on $L$ to update $\theta$

