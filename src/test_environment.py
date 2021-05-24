import envs, gym
gym.logger.set_level(40)
import safety_gym
import numpy as np

env = gym.make('Safe-Half-Cheetah-v0')
env.reset()
print(env.rm_state_features)
for _ in range(100):
	env.render()
	action = env.action_space.sample()
	observation, reward, cost, done, info = env.step(action)
	if done:
		print("Episode finished after {} timesteps".format(_+1))
		break
env.close()