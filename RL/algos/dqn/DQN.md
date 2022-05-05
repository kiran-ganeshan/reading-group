# Deep Q-Network
Requirements: [[TQL]], [[Neural Network]]

## Resources
- The original paper, [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Papers with code](https://paperswithcode.com/method/dqn#:~:text=A%20DQN%2C%20or%20Deep%20Q,framework%20with%20a%20neural%20network.&text=It%20is%20usually%20used%20in%20conjunction%20with%20Experience%20Replay%2C%20for,the%20replay%20memory%20at%20random.)
- [SpinningUp Deep Q-Learning Reading List](https://spinningup.openai.com/en/latest/spinningup/keypapers.html#a-deep-q-learning) (in particular, Double Q-Learning for preventing optimistic Q-values)
- Some problems with DQN described in [Rainbow](https://arxiv.org/abs/1710.02298)

## Introduction
You've seen how [[TQL]] can be used to solve MDPs. Recall that our $Q$ function is represented as an array/table of size $|S \times A|$. The larger our state and/or action space, the costlier this representation becomes. 

The core idea behing a Deep Q-network is to represent the $Q$ function with a function approximator, a deep neural network in this case. The associated algorithm, deep Q-learning, trains the deep neural network to output $Q^*(s, a)$ for any input state-action pair. This allows us to extract the optimal policy in a discrete, finite action space, since $\pi^*(a \mid s) = \argmax_{a\in A} Q^*(s, a)$.

## Strategy

As in TQL, we will use the [[Value Functions#Bellman Recurrence | Bellman Recurrence]]  to estimate Q-values:
$$Q(s, a) = r(s, a) + \gamma\E_{s' \sim \mathcal{T}(s' \mid s, a)}\max_{a'\in A} Q(s', a')$$
Just as in TQL, once we have the optimal $Q^*$ values over a discrete action space, we can extract the policy using the argmax:
$$\pi(a\mid s) = \begin{cases}1 &\text{if }a = \argmax_{a\in A}Q(s, a)\\0 &\text{otherwise}\end{cases}$$
(If multiple actions maximize $Q(s, a)$, we consider $\arg\max$ to output the *first* maximizing action in the action space given some arbitrary ordering of actions.)

In TQL, we computed the maximum over the action space directly (working only with finite discrete action spaces), and we approximated the expectation over the transition by averaging the results of sampling $s'\sim \mathcal{T}(s'\mid s,a)$ and evaluating $\max_{a'\in A}Q(s', a')$. Here's how both of these are accomplished in the DQN:

#### Computing the Maximum

When approximating $Q$-values with a NN, the first approach that comes to mind is to simply concatenate state-action pairs and map them to $Q$-values:

![](Images/Q(s,a).png)

However, due to the Bellman Recurrence we have to evaluate $\max_{a'\in A}Q(s', a')$ to obtain our target values, which takes one evaluation of the NN for each action in the action space. For large finite action spaces, this can become intractible. Therefore we make our NN take the state $s$ and output a vector of $Q$-values $Q(s, a_i)$ for each $a_i\in A$:

![](Images/Q(s,cdot).png)

Then, to compute the maximum, we need only compute a single forward pass and then take the maximum over the output $Q$-values.

#### Approximating the Expectation
Consider a Neural Network $f: \mathcal{X}\to\mathcal{Y}$ mapping a space $\mathcal{X}$ into a space $\mathcal{Y}$. Say we train the NN with an MSE Loss on the training dataset $\{(x, y_1), (x, y_2), \dots, (x, y_n)\}$ where each training point has the same input but different outputs. When we train the NN on the full dataset, the loss function looks like
$$\frac{1}{n}\Bigg[\sum_{i=1}^n(f(x) - y_i)^2\Bigg]$$
This loss function is minimized by the average of the target values:
$$f^*(x) = \frac{1}{n}\sum_{i=1}^n y_i$$
so the NN will naturally train to output the average of the conflicting targets we assign to $x$.

In DQN, we take advantage of this to avoid having the employ the continual averaging tricks we used in TQL. In TQL, in order to approximate the expectation over states in the Bellman recurrence, we stored and continually updated the running average of the target values for $\hat{Q}(s, a)$. When using a neural network based DQN, the above property of MSE loss makes this simpler. We simply add the data point $(s, a, r(s, a), s')$ to our replay buffer for each transition from state $s$ to state $s'$ via action $a$. We then train our neural network to the targets
$$\Big((s, a), \;r(s, a) + \gamma\max_{a'\in A}Q(s', a')\Big)$$
If we are training with MSE loss, given that $s' \sim \mathcal{T}(s'\mid s, a)$, this will train our neural network to output
$$Q(s, a) =\E_{s'\sim\mathcal{T}(s'\mid s, a)}\Big[r(s, a) + \gamma\max_{a'\in A} Q(s', a')\Big]$$
so we will be training our neural network to satisfy the Bellman recurrence of the optimal $Q$-function.

#### Tricks

A few tricks are necessary in order to get DQN to work.
1. Replay Buffer: NNs (theoretically and experimentally) converge on Independent and Identically Distributed (I.I.D.) samples, but the state transitions $(s, a, s', r)$ that we experience while rolling out the algorithm are highly correlated. Hence, we store all the state transitions we experience in a **replay buffer** and train using batches of transitions sampled from the replay buffer:
$$\mathcal{R} = \{(s_0, a_0, s_1, r(s_0, a_0)),(s_1, a_1, s_2, r(s_1, a_1)), \dots, \} $$

2. Target Network: Notice that the target value we use to train the NN, namely $r(s, a) + \max_{a'\in A}Q(s', a')$, depend on the output of the NN on the state-action pairs $(s', a')$. This can lead to unstable training, so we introduce a stable copy of the $Q$-network called the **target network**. We will call the regular $Q$-network parameters $\theta$ and the target $Q$-network parameters $\theta'$. (The target $Q$-function is called $Q'(s, a)$.) Each time we take a gradient step, we update the target network using the exponential moving average: $\theta'\leftarrow (1 - \tau)\theta' + \tau\theta$. The target network then tracks the current $Q$-values while being robust to short-term variation. We use this copy to
	1. Evaluate the target values
	2. Take actions in the environment when generating transitions

3. $\varepsilon$-greedy Exploration: We mentioned that when evaluating a trained DQN, we take actions corresponding to the $\arg\max$ of the $Q$-values. However, if our algorithm never explores certain choices of actions, it may never discover that alternative actions actually achieve higher $Q$-value. Therefore, we use $\varepsilon$-greedy Exploration, in which our agent chooses the $Q$-value-based action with probability $1 - \varepsilon$ and chooses a random action with probability $\varepsilon$. This works out to the following distribution:
$$\pi(a\mid s) = \begin{cases}1-\varepsilon + \frac{\varepsilon}{|A|} &\text{if }a = \argmax_{a\in A}Q'(s, a)\\\frac{\varepsilon}{|A|} &\text{otherwise}\end{cases}$$

## Use Cases
DQN can be used to solve MDPs with 
- Discrete or Continuous State Spaces
- Discrete and Finite Action Spaces

## Algorithm

**DQN**: Deep Q-Learning with Experience Replay
**Parameters**: 
- Number of total steps $T$
- Replay Buffer capacity $N$
- Exploration parameter $\ep$

**Algorithm**:
> Initialize replay buffer $\mathcal{D}$ with capacity $N$\
> Initialize $Q$-function neural network with random weights $\theta$\
> Initialize environment
> **for** step $\in\{1, \dots, T\}$ **do**:\
> $\qquad$ With probability $\ep$: \
> $\qquad\qquad$ Select a random action $a$\
> $\qquad$ Otherwise: \
> $\qquad\qquad$ Select $a = \max_{a\in A}Q_\theta(s_t, a)$\
> $\qquad$ Execute action $a$, retrieve reward $r$ and next state $s'$\
> $\qquad\qquad$ Store transition $(s_t, a_t, r_t, s_{t+1})$ in $\mathcal{D}$\
> $\qquad\qquad$ Sample random batch of transitions $(s_j, a_j, r_j, s_{j+1})$ from $\mathcal{D}$\
> $\qquad\qquad$ Set $y_j = \begin{cases} r_j & \text{if }s_{j+1}\text{ is terminal} \\ r_j + \gamma \max_{a'\in A}Q_\theta(s_{j+1}, a') & \text{if }s_{j+1}\text{ is not terminal}\end{cases}$\
> $\qquad\qquad$ Set our loss to $L = \sum_{j} \left(y_j - Q_\theta(s_j,a_j )\right)^2$\
> $\qquad\qquad$ Perform gradient descent on $L$ to update $\theta$




