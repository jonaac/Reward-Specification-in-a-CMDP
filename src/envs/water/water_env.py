import gym
from gym import spaces
import numpy as np
from rm.rm_env import RewardMachineEnv
from envs.water.water_world import WaterWorld, WaterWorldParams

class WaterEnv(gym.Env):
	def __init__(self, state_file):
		params = WaterWorldParams(
			state_file, b_radius=15, max_x=400, 
			max_y=400, b_num_per_color=2, use_velocities=True
		)
		self.params = params

		self.action_space = spaces.Discrete(5) # noop, up, right, down, left
		self.observation_space = spaces.Box(low=-2, high=2, 
											shape=(52,), dtype=np.float)
		self.env = WaterWorld(params)

	def get_events(self):
		return self.env.get_true_propositions()

	def step(self, action):
		self.env.execute_action(action)
		obs = self.env.get_features()
		reward = 0 # all the reward comes from the RM
		cost = 0
		done = False
		info = {}
		return obs, reward, done, info

	def reset(self):
		self.env.reset()
		return self.env.get_features()

# SINGLE TASK --------------------------------------------------------------------------------------------------


class WaterEnvRM(RewardMachineEnv):
	def __init__(self):
		env = WaterEnv(None)
		rm_files = ["./envs/water/rm/rm.txt"]
		super().__init__(env, rm_files)

	def render(self, mode='human'):
		if mode == 'human':
			# add play(self) to water_world.py
			raise NotImplementedError
		else:
			raise NotImplementedError
