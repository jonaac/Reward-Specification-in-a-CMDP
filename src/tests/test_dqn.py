import time
import sys
import psutil

import envs
import gym
import rl

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from rl.disc.dqn import DQL
from rl.parameters import (CHECKPOINTS_PATH, TOTAL_EPISODES, 
							TF_LOG_DIR, UNBALANCE_P)

gym.logger.set_level(40)
SAVE_WEIGHTS = True

env = gym.make('Safe-Water-World-NoMachine-v0')
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

dql = DQL(observation_space, action_space)

# weighted sum of rewards for each epoch
acc_reward = tf.keras.metrics.Sum('reward',dtype=tf.float32)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []


EPOCHS = 1000
T = 600

print('Start Training')
print('--------------')

for epoch in range(EPOCHS):
	s = env.reset()
	acc_reward.reset_states()
	dql.noise.reset()
	print('Epoch #' + str(epoch))
	time_load = []
	time_train = []

	for t in range(T):
		a, q_l = dql.act(s)
		sn, r, _, done, info = env.step(a)
		dql.remember(s,r,sn,int(done))

		if dql.can_train():
			entry = dql.buffer.get_batch(unbalance_p=UNBALANCE_P)
			q = dql.learn(entry)

		acc_reward(r)
		s = sn

		if done:
			print("Episode finished after {} timesteps".format(t+1))
			break

	# print(psutil.virtual_memory().percent)

	ep_reward_list.append(acc_reward.result().numpy())
	# Mean of reward obtained in last 40 steps fo training
	if epoch >= 40:
		avg_reward = np.mean(ep_reward_list[-40:])
		print('Average Reward of last 40 steps: ' + str(avg_reward))
		# add to list to plot
		avg_reward_list.append(avg_reward)

	# save weights
	if (epoch % 5 == 0 and
		SAVE_WEIGHTS and
		epoch >= 5):

		dql.save_weights(CHECKPOINTS_PATH)

env.close()
dql.save_weights(CHECKPOINTS_PATH)

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()