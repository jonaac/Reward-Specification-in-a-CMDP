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
		for rm_file, cm_file in rm_files:
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

		# Computing one-hot encodings for the non-terminal safe RM states
		self.rm_state_features = {}
		for rm_id, rm in enumerate(self.reward_machines):
			for u_id in rm.get_states():
				u_features = np.zeros(self.num_rm_states)
				u_features[len(self.rm_state_features)] = 1
				self.rm_state_features[(rm_id,u_id)] = u_features
		# for terminal RM states, we give as features an array of zeros
		self.rm_done_feat = np.zeros(self.num_rm_states)

		# Selecting the current RM task
		self.current_rm_id = -1
		self.current_rm	= None

	def reset(self):
		# Reseting the environment and selecting the next RM tasks
		self.obs = self.env.reset()
		self.current_rm_id = (self.current_rm_id+1)%self.num_rms
		self.current_rm	= self.reward_machines[self.current_rm_id]
		self.current_u_id  = self.current_rm.reset()

		# Adding the RM state to the observation
		return self.get_observation(self.obs, self.current_rm_id, 
									self.current_u_id, False)

	def step(self, action):
		# executing the action in the environment
		next_obs, original_reward, env_done, info = self.env.step(action)

		# getting the output of the detectors and saving information for 
		# generating counterfactual experiences
		true_props = self.env.get_events()
		if true_props != '': 
			#print(true_props)
			pass

		self.crm_params = self.obs, action, next_obs, env_done, true_props, info
		self.obs = next_obs

		old_u_id = self.current_u_id

		# update the RM state
		self.current_u_id, rm_r, rm_c, rm_done = self.current_rm.step(
													self.current_u_id, 
													true_props,
													info
												)

		#print("{} -> {}".format(old_u_id,self.current_u_id))

		# returning the result of this action
		done = rm_done or env_done
		rm_obs = self.get_observation(
			next_obs, 
			self.current_rm_id,
			self.current_u_id,
			done
		)

		return rm_obs, rm_r, rm_c, done, info

	def get_observation(self, next_obs, rm_id, u_id, done):
		if done:
			rm_feat = self.rm_done_feat
		else:
			rm_feat = self.rm_state_features[(rm_id,u_id)]
		rm_obs = {'features': next_obs,'rm-state': rm_feat}
		return gym.spaces.flatten(self.observation_dict, rm_obs)
