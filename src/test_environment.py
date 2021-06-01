import envs, gym
gym.logger.set_level(40)
import safety_gym
import numpy as np

env = gym.make('Safe-Water-World-v0')
env.reset()
env.who()
env.close()