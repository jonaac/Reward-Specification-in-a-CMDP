import time
import sys
from argparse import ArgumentParser

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from rl.disc.dqn import DQL
from rl.dqn_parameters import (
	CHECKPOINTS_PATH, TOTAL_STEPS,
	TF_LOG_DIR, UNBALANCE_P)


def main():

	parser = ArgumentParser(
		description="Deep Q-Learning with Safety Machines"
	)
	parser.add_argument('--env', type=str, nargs='?',
						default="Safe-Water-World-NoMachine-v0",
						help='The OpenAI Gym environment to train on, '
							 'e.g. Safe-Water-World-NoMachine-v0, Safe-Water-World-RewardMachine-v0'
							 ' Safe-Water-World-SafetyMachine-v0')
	parser.add_argument('--crm', type=bool, nargs='?', default=False,
						help='Add reward machine/safety machine states to replay buffer')
	parser.add_argument('--save_weights', type=bool, nargs='?', default=True,
						help='Save the weight of the network in the defined checkpoint file '
							 'directory.')

	args = parser.parse_args()
	RL_TASK = args.env
	SAVE_WEIGHTS = args.save_weights
	CRM = args.crm

	env = gym.make(RL_TASK)
	observation_space = env.observation_space.shape[0]
	action_space = env.action_space.n

	dql = DQL(observation_space, action_space)

	s = env.reset()

	# weighted sum of rewards for each epoch
	acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
	acc_cost = tf.keras.metrics.Sum('cost', dtype=tf.float32)

	# To store reward history of each episode
	ep_reward_list = []
	ep_cost_list = []
	# To store average reward history of last few episodes
	avg_reward_list = []
	avg_cost_list = []

	print('* ----- Start Training ----- ')
	print('-----------------------------')

	step_count, epoch_count = 0, 0
	for t in TOTAL_STEPS:
		step_count += 1
		a, q_l = dql.act(s)
		sn, r, d, done, info = env.step(a)

		if CRM:
			experiences = info['crm-experience']
			dql.remember(experiences)
		else:
			dql.remember(s, a, r, d, sn, int(done))

		if dql.can_train():
			entry = dql.buffer.get_batch(unbalance_p=UNBALANCE_P)
			q = dql.learn(entry)

		if done:
			print("Episode {} finished after {} time-steps".format(epoch_count, step_count))
			print("Currently at {} total time-steps".format(epoch_count, step_count))
			print('-----------------------------')
			s = env.reset()
			acc_reward.reset_states()
			acc_cost.reset_states()
			epoch_count += 1
			step_count = 0

		ep_reward_list.append(acc_reward.result().numpy())
		# Mean of reward obtained in last 40 steps fo training
		if epoch_count >= 40:
			avg_reward = np.mean(ep_reward_list[-40:])
			# add to list to plot
			avg_reward_list.append(avg_reward)

		# save weights
		if (epoch_count % 100 == 0 and
			epoch_count >= 100 and
			SAVE_WEIGHTS):
			dql.save_weights(CHECKPOINTS_PATH)

	env.close()
	dql.save_weights(CHECKPOINTS_PATH)

	plt.plot(avg_reward_list)
	plt.xlabel("Episode")
	plt.ylabel("Avg. Epsiodic Reward")
	plt.show()


if __name__ == '__main__':
	main()
