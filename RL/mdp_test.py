from filename import Env
import numpy as np
env = Env(
    size=10,
    layout=[8 * [8 * [0]], [1, 0] + 8 * [0], [0, 1] + 8 * [0]],
    discrete=False,
    max_distance=0.5,
    goal_distance=0.25
)
state = env.reset([.9, .9])
state, action, done, info = env.step([1, 1])
assert state == np.array([.1, .1])