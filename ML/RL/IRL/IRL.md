# Inverse Reinforcement Learning
#todo
The following table summarizes the differences between traditional and inverse RL:

| Traditional RL | Inverse RL |
| --- | --- |
| Given: <ul><li>States space $S$</li><li>Action space $A$</li><li>(Sometimes) transitions $\mathcal{T}(s'\mid s, a)$</li> <li>Reward function $r(s, a)$</li> </ul> | Given: <ul><li>States space $S$</li><li>Action space $A$</li><li>(Sometimes) transitions $\mathcal{T}(s'\mid s, a)$</li> <li>Trajectories $\{\tau_i\}$ sampled from the optimal policy $\pi^*(a\mid s)$</li></ul> |
| Learning: <ul><li>The optimal policy $\pi^*_\theta(a\mid s)$</li></ul> | Learning: <ul><li>The reward function $r_\phi(s, a)$</li></ul> |


