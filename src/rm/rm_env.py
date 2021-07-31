import gym
from gym import spaces
import numpy as np
from rm.rm import SafetyMachine


class RewardMachineEnv(gym.Wrapper):
	def __init__(self, env, sm_files):
		super().__init__(env)

		# Loading the reward machines
		self.sm_files = sm_files
		self.safety_machines = [] # list of safety machines
		self.num_rm_states = 0 # number of total rm states
		self.num_cm_states = 0 # number of total cm states
		for rm_file, cm_file in sm_files:
			sm = SafetyMachine(rm_file, cm_file)
			self.num_rm_states += len(sm.get_rm_states())
			self.num_cm_states += len(sm.get_cm_states())
			self.safety_machines.append(sm)
		self.num_sm = len(self.safety_machines) # number safety machines

		# The observation space is a dictionary including the env features and 
		# a one-hot representation of the state in the reward machine
		feat = env.observation_space
		rm_s = spaces.Box(
			low=0,
			high=1, 
			shape=(self.num_rm_states,), 
			dtype=np.uint8
		)
		cm_s = spaces.Box(
			low=0,
			high=1, 
			shape=(self.num_cm_states,), 
			dtype=np.uint8
		)
		self.observation_dict  = spaces.Dict({
									'features':feat,
									'rm-state':rm_s,
									'cm-state':cm_s
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

		self.rm_state_features = {}
		self.cm_state_features = {}
		for sm_id, sm in enumerate(self.safety_machines):

			for u_id in sm.rm.get_states():
				u_features = np.zeros(self.num_rm_states)
				u_features[len(self.rm_state_features)] = 1
				self.rm_state_features[(sm_id,u_id)] = u_features

			for u_id in sm.cm.get_states():
				u_features = np.zeros(self.num_cm_states)
				u_features[len(self.cm_state_features)] = 1
				self.cm_state_features[(sm_id,u_id)] = u_features
		
		self.rm_done_feat = np.zeros(self.num_rm_states)
		self.cm_done_feat = np.zeros(self.num_cm_states)

		# Selecting the current RM task
		self.current_sm_id = -1
		self.current_sm	= None

	def reset(self):
		# Reseting the environment and selecting the next RM tasks
		self.obs = self.env.reset()
		self.current_sm_id = (self.current_sm_id+1)%self.num_sm
		self.current_sm = self.safety_machines[self.current_sm_id]
		self.current_rm_u_id, current_cm_u_id = self.current_sm.reset()

		# Adding the RM state to the observation
		return self.get_observation(self.obs, 
									self.current_sm_id,
									self.current_rm_u_id
									self.current_cm_u_id,
									False)

	def step(self, action):
		# executing the action in the environment
		next_obs, original_reward, env_done, info = self.env.step(action)

		# getting the output of the detectors and saving information for 
		# generating counterfactual experiences
		true_props = self.env.get_events()

		self.crm_params = self.obs, action, next_obs, env_done, true_props, info
		self.obs = next_obs

		old_u_id = self.current_u_id

		# update the RM state
		self.current_rm_u_id, r, rm_done = self.current_sm.rm.step(
			self.current_rm_u_id, 
			true_props,
			info
		)

		self.current_cm_u_id, d, cm_done = self.current_sm.cm.step(
			self.current_cm_u_id, 
			true_props,
			info
		)

		# returning the result of this action
		done = rm_done or env_done or cm_done
		rm_obs = self.get_observation(
			next_obs, 
			self.current_rm_id,
			self.current_u_id,
			done
		)

		return rm_obs, r, d, done, info

	def get_observation(self, next_obs, sm_id, rm_u_id, cm_u_id, done):
		if done:
			rm_feat = self.sm_done_feat
			cm_feat = self.cm_done_feat
		else:
			rm_feat = self.rm_state_features[(sm_id,rm_u_id)]
			cm_feat = self.cm_state_features[(sm_id,cm_u_id)]
		
		sm_obs = {
			'features': next_obs,
			'rm-state': rm_feat,
			'cm-state': cm_feat
		}

		return gym.spaces.flatten(self.observation_dict, sm_obs)
