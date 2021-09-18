import argparse
import logging
import random

import gym, safety_gym
gym.logger.set_level(40)
from tqdm import trange
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# <----- NEDS TO BE DEVELOPED -------> #
'''
from rm import RewardMachine
'''
from clrm import CLRM # Not Finished
from parameters import (CHECKPOINTS_PATH, TOTAL_EPISODES,
								TF_LOG_DIR, UNBALANCE_P)
from utils import Tensorboard

if __name__ == "__main__":

	logging.basicConfig()
	logging.getLogger().setLevel(logging.INFO)

	parser = argparse.ArgumentParser(
		prog="Deep Deterministic Policy Gradient (DDPG)",
		description="Deep Deterministic Policy Gradient (DDPG) in Tensorflow 2"
	)

	parser.add_argument('--env', type = str, nargs = '?',
						default = 'Safexp-PointPush1-v0',
						help = 	'The Safety Gym environment to train on, '
								'e.g. Safexp-PointPush1-v0, ' 
								'Safexp-DoggoGoal2-v0, '
								'Safexp-CarButton1-v0')
	parser.add_argument('--render_env', type = bool, 
						nargs = '?', default = False,
						help = 'Render the environment to be visually visible')
	parser.add_argument('--train', type = bool,
						nargs = '?', default = True,
						help = 'Train the network on the modified DDPG algorithm')
	parser.add_argument('--use_noise', type = bool,
						nargs = '?', default = True,
						help = 'OU Noise will be applied to the policy action')
	parser.add_argument('--eps_greedy', type = float,
						nargs = '?', default = 0.95,
						help = 	'The epsilon for Epsilon-greedy '
								'in the policy\'s action')
	parser.add_argument('--warm_up', type = bool, 
						nargs = '?', default = 1,
						help = 	'Following recommendation from OpenAI'
								'Spinning Up, the actions in the early epochs'
								'can be set random to increase exploration.' 
								'This warm up defines how many epochs are' 
								'initially set to do this.')
	parser.add_argument('--save_weights', type = bool,
						nargs = '?', default = True,
						help = 'Save the weight of the network in the defined'
						'checkpoint file directory.')

	'''
	SET PARAMETERS
	'''

	args = parser.parse_args()
	SAFE_TASK = args.env
	RENDER_ENV = args.render_env
	LEARN = args.train
	USE_NOISE = args.use_noise
	WARM_UP = args.warm_up
	SAVE_WEIGHTS = args.save_weights
	EPS_GREEDY = args.eps_greedy

	'''
	INITIALIZE ENVIRONMENT
	'''
	env = gym.make(SAFE_TASK)
	action_space_high = env.action_space.high[0]
	action_space_low = env.action_space.low[0]
	observation_space = env.observation_space.shape[0]
	action_space = env.action_space.shape[0]

	'''
	INITIALIZE REWARD MACHINE
	'''

	'''
	INITIALIZE LEARNER
	'''
	clrm = CLRM(observation_space, action_space,
				action_space_high, action_space_low)
	tensorboard = Tensorboard(log_dir=TF_LOG_DIR)

	# load weights if available
	logging.info("Loading weights from %s*", CHECKPOINTS_PATH)
	clrm.load_weights(CHECKPOINTS_PATH)

	'''
	METRICS
	'''
	# weighted sum of rewards for each epoch
	acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
	# mean of critic and actor loss values
	Q_loss = tf.keras.metrics.Mean('Q_loss', dtype=tf.float32)
	A_loss = tf.keras.metrics.Mean('A_loss', dtype=tf.float32)

	# To store reward history of each episode
	ep_reward_list = []
	# To store average reward history of last few episodes
	avg_reward_list = []

	'''
	TRAIN AGENT
	'''
	with trange(TOTAL_EPISODES) as t:
		for ep in t:
			prev_state = env.reset()
			acc_reward.reset_states()
			Q_loss.reset_states()
			A_loss.reset_states()
			clrm.noise.reset()

			for _ in range(2000):

				eps_bool = EPS_GREEDY+(1-EPS_GREEDY)*ep/TOTAL_EPISODES
				
				# Recieve state and reward from environment.
				cur_act = clrm.act(tf.expand_dims(prev_state, 0),
									_notrandom=(ep >= WARM_UP)
									and (random.random() < eps_bool),
									noise=USE_NOISE)
				state, reward, done, _ = env.step(cur_act)
				clrm.remember(prev_state, reward, state, int(done))

				# Update weights
				if LEARN:
					c, a = clrm.learn(clrm.get_batch(UNBALANCE_P))
					Q_loss(c)
					A_loss(a)

				# Add reward from this step
				acc_reward(reward) 
				prev_state = state # update states

				if done:
					break

			ep_reward_list.append(acc_reward.result().numpy())
			# Mean of reward obtained in last 40 steps fo training
			avg_reward = np.mean(ep_reward_list[-40:])
			# add to list to plot
			avg_reward_list.append(avg_reward)

			# save weights
			if ep % 5 == 0 and SAVE_WEIGHTS:
				clrm.save_weights(CHECKPOINTS_PATH)

	env.close()
	clrm.save_weights(CHECKPOINTS_PATH)
	logging.info("Training done...")

	'''
	Plot rewards, costs and other evaluations
	'''
	plt.plot(avg_reward_list)
	plt.xlabel("Episode")
	plt.ylabel("Avg. Epsiodic Reward")
	# Plotting graph, Episodes versus Avg. Rewards
	plt.show()

	pass