import gym
import numpy as np
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from rm.rm_env import RewardMachineEnv

class HalfCheetahSafeEnv(gym.Wrapper):
	def __init__(self):
		# Note that the current position is key for our tasks
		super().__init__(HalfCheetahEnv(
			exclude_current_positions_from_observation=False
		))

	def step(self, action):
		# executing the action in the environment
		next_obs, original_reward, env_done, info = self.env.step(action)
		self.info = info
		return next_obs, original_reward, env_done, info

	def get_events(self):
		events = ''
		if self.info['x_position'] < -10:
			events += 'b'
		if self.info['x_position'] > 10:
			events += 'a'
		if self.info['x_position'] < -2:
			events += 'd'
		if self.info['x_position'] > 2:
			events += 'c'
		if self.info['x_velocity'] > 1:
			events += 'v'
		return events

class HalfCheetahSafeEnvRM(RewardMachineEnv):
	def __init__(self):
		env = HalfCheetahSafeEnv()
		rm_files = ["./envs/halfcheetah/rm/rm.txt"]
		super().__init__(env, rm_files)