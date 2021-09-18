import envs
import gym
import rl

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from rl.cont.sddpg import SDDPG

observation_space = 10
action_space = 4
sddpg = SDDPG(observation_space, action_space, 1)

s = np.random.rand(observation_space,)
sddpg.act(s)