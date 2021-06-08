import time
import sys
import psutil

import envs, gym, rl
gym.logger.set_level(40)

import numpy as np
import tensorflow as tf

from rl.disc.dqn import DQL
from rl.parameters import (CHECKPOINTS_PATH, TOTAL_EPISODES, 
							TF_LOG_DIR, UNBALANCE_P)

SAVE_WEIGHTS = True

env = gym.make('Safe-Water-World-v0')
observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

dql = DQL(observation_space, action_space)

# weighted sum of rewards for each epoch
acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
# mean of critic and actor loss values
Q_loss = tf.keras.metrics.Mean('Q_loss', dtype=tf.float32)
A_loss = tf.keras.metrics.Mean('A_loss', dtype=tf.float32)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []


EPOCHS = 100
T = 600

print('Start Training')
print('--------------')

for epoch in range(EPOCHS):
	s = env.reset()
	acc_reward.reset_states()
	dql.noise.reset()
	start_time = time.time()
	print('Epoch #' + str(epoch))

	for i in range(T):
		a, q_l = dql.act(s)
		sn, r, _, done, info = env.step(a)
		brain = dql.remember(s,r,sn,int(done))

		if dql.can_train():
			entry = dql.buffer.get_batch(unbalance_p=UNBALANCE_P)
			q = dql.learn(entry)

		if done:
			break

		acc_reward(r)
		s = sn

	print(psutil.virtual_memory().percent)

	ep_reward_list.append(acc_reward.result().numpy())
	# Mean of reward obtained in last 40 steps fo training
	avg_reward = np.mean(ep_reward_list[-40:])
	print('Average Reward of last 40 steps: ' + str(avg_reward))
	# add to list to plot
	avg_reward_list.append(avg_reward)

	print("--- %s seconds ---" % (time.time() - start_time))

	# save weights
	if (epoch % 5 == 0 and
		SAVE_WEIGHTS and
		epoch >= 5):

		dqn.save_weights(CHECKPOINTS_PATH)

env.close()
dqn.save_weights(CHECKPOINTS_PATH)
logging.info("Training done...")

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
# Plotting graph, Episodes versus Avg. Rewards
plt.show()

