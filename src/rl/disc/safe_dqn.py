import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from rl.parameters import (KERNEL_INITIALIZER, GAMMA, RHO, STD_DEV, 
						BUFFER_SIZE, BATCH_SIZE, Q_LR, COST_BOUND)
from rl.utils import OUActionNoise, ReplayBuffer
from rl.disc.dqn import QNetwork, DQN

class SDQN(DQN):

	def __init__(
			self, num_states, num_actions,
			gamma=GAMMA, rho=RHO, std_dev=STD_DEV):

		super().__init__(
			num_states, num_actions, action_high,
			GAMMA, RHO, STD_DEV)

		self.cost_network = QNetwork(num_states, num_actions)
		self.cost_target = QNetwork(num_states, num_actions)

		self.cost_optimizer = tf.keras.optimizers.Adam(Q_LR, amsgrad=True)

		# define update weights
		@tf.function(input_signature=[
			tf.TensorSpec(shape=(None, self.num_states), dtype=tf.float32),
			tf.TensorSpec(shape=(None, self.num_actions), dtype=tf.int64),
			tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
			tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
			tf.TensorSpec(shape=(None, self.num_states), dtype=tf.float32),
			tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
		])
		def update_weights(s, a, r, d, sn, done):
			test = [s, a, r, d, sn, done]
			# ---------------------------------------------------------------------------- #
			# Update Q-Network
			with tf.GradientTape() as tape:
				tape.watch(self.dqn.trainable_variables)
				
				# Compute y
				q_values = self.dqn_target(sn)
				an = tf.argmax(q_values, axis=1)
				
				one_hot = tf.one_hot(an, self.num_actions)
				q_value = one_hot * q_values
				q_value = tf.reduce_sum(q_value, axis = 1)
				y = r + (1 - done) * self.gamma * q_value

				# Compute Q(s,a)
				q_values = self.dqn(s)
				one_hot = tf.one_hot(a, self.num_actions)
				q_value = one_hot * q_values
				q_value = tf.reduce_sum(q_value, axis = 1)

				# Compute error
				error = tf.math.abs(y - q_value)
				q_loss = tf.math.reduce_mean(error)

			q_grad = tape.gradient(
				q_loss,
				self.dqn.trainable_variables
			)
			self.q_optimizer.apply_gradients(
				zip(q_grad, self.dqn.trainable_variables)
			)
			# ---------------------------------------------------------------------------- #
			# Update Cost Network
			with tf.GradientTape() as tape:
				tape.watch(self.cost_network.trainable_variables)
				
				# Compute y
				cost_values = self.cost_target(sn) 
				an = tf.argmax(cost_values, axis=1)
				
				one_hot = tf.one_hot(an, self.num_actions)
				cost_value = one_hot * cost_values
				cost_value = tf.reduce_sum(cost_value, axis = 1)
				z = d + (1 - done) * self.gamma * cost_value

				# Compute J(s,a)
				cost_values = self.dqn(s)
				one_hot = tf.one_hot(a, self.num_actions)
				cost_value = one_hot * cost_values
				cost_value = tf.reduce_sum(cost_value, axis = 1)

				# Compute error
				error = tf.math.abs(z - cost_value)
				cost_loss = tf.math.reduce_mean(error)

			cost_grad = tape.gradient(
				cost_loss,
				self.dqn.trainable_variables
			)
			self.cost_optimizer.apply_gradients(
				zip(cost_grad, self.cost_network.trainable_variables)
			)
			
			return q_loss

	def act(self, state, _notrandom=True):
		state = tf.convert_to_tensor([state], dtype=tf.float32)
		q_value = self.dqn(state)[0].numpy()
		cost_value = self.cost_network(state)[0].numpy()

		safe_bool = cost_value < COST_BOUND
		safe_value = q_value * safe_bool

		is_all_zero = np.all((safe_value == 0))

		if _notrandom:
			if is_all_zero:
				self.cur_action = np.argmax(safe_value)
			else:
				self.cur_action = np.argmax(q_value)
		else:
			self.cur_action = np.random.choice(self.num_actions)
		
		return self.cur_action, q_value

	def learn(self, entry):
		s, a, r, _, sn, done = zip(*entry)

		q_l = self.update_weights(	
			tf.convert_to_tensor(s,dtype=tf.float32),
			tf.convert_to_tensor(a,dtype=tf.int64),
			tf.convert_to_tensor(r,dtype=tf.float32),
			tf.convert_to_tensor(sn,dtype=tf.float32),
			tf.convert_to_tensor(done,dtype=tf.float32)
		)

		self.update_target(self.dqn_target, self.dqn, self.rho)
		self.update_target(self.cost_target, self.cost_network, self.rho)

		return q_l
