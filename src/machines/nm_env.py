import gym
from gym import spaces
import numpy as np
from machines.machines import SafetyMachine


class NoMachineEnv(gym.Wrapper):
	
	def __init__(self, env, rm_files, cm_files):
		super().__init__(env)

		# Loading the reward machines
		self.safety_machines = [] # list of safety machines
		self.num_rm_states = 0 # number of total rm states
		self.num_cm_states = 0 # number of total cm states
		for rm_file, cm_file in zip(rm_files, cm_files):
			sm = SafetyMachine(rm_file, cm_file)
			self.num_rm_states += len(sm.get_rm_states())
			self.num_cm_states += len(sm.get_cm_states())
			self.safety_machines.append(sm)
		self.num_sm = len(self.safety_machines)

		# The observation space is a dictionary including the env features and 
		# a one-hot representation of the state in the reward machine
		feat = env.observation_space

		self.observation_dict  = spaces.Dict({
									'features':feat
								})

		flatdim = gym.spaces.flatdim(self.observation_dict)
		s_low  = float(env.observation_space.low[0])
		s_high = float(env.observation_space.high[0])
		self.observation_space = spaces.Box(
			low=s_low,
			high=s_high,
			shape=(flatdim,), 
			dtype=np.float32
		)

		# Selecting the current RM task
		self.current_sm_id = -1
		self.current_sm	= None

	def reset(self):
		# Reseting the environment and selecting the next RM tasks
		self.obs = self.env.reset()
		self.current_sm_id = (self.current_sm_id+1)%self.num_sm
		self.current_sm = self.safety_machines[self.current_sm_id]
		self.current_rm_u_id, self.current_cm_u_id = self.current_sm.reset()

		obs = {
			'features': next_obs,
		}
		return gym.spaces.flatten(self.observation_dict, obs)

	def step(self, action):
		# executing the action in the environment
		next_obs, original_reward, env_done, info = self.env.step(action)

		# getting the output of the detectors and saving information for 
		# generating counterfactual experiences
		true_props = self.env.get_events()
		self.obs = next_obs

		# update the RM state
		self.current_rm_u_id, r, rm_done = self.current_rm.step(
			self.current_rm_u_id, 
			true_props,
			info)

		self.current_cm_u_id, d, cm_done = self.current_sm.cm.step(
			self.current_cm_u_id, 
			true_props,
			info)

		# returning the result of this action
		done = rm_done or env_done or cm_done
		obs = {
			'features': next_obs,
		}
		obs = gym.spaces.flatten(self.observation_dict, obs)

		return obs, r, d, done, info
		