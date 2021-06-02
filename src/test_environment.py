import time

import envs, gym, rl
gym.logger.set_level(40)
import safety_gym

import numpy as np
import tensorflow as tf

from rl.disc.dqn import DQL

dqn = DQL(5,5,1,-1)

states = []
actions = []
rewards = []
t = False
for i in range(1000):
	if not t:
		s = np.random.uniform(0,1,dqn.num_states)
		t = True
	else:
		s = sn
	dqn.cur_action = np.random.choice(dqn.num_actions)
	r = np.random.choice([0,1])
	sn = np.random.uniform(0,1,dqn.num_states)
	dqn.remember(s,r,sn,0)

for i in range(1):
	q = dqn.learn(dqn.buffer.get_batch(unbalance_p=0.8))
	'''
	r = tf.one_hot(a, dqn.num_actions)
	tf.print(r)
	t = r * q
	tf.print(t)
	u = tf.reduce_sum(t, axis = 1)
	tf.print(u)'''