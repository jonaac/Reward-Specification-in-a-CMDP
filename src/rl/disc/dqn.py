import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from rl.parameters import (KERNEL_INITIALIZER, GAMMA, RHO, STD_DEV, 
						BUFFER_SIZE, BATCH_SIZE, Q_LR)
from rl.utils import OUActionNoise, ReplayBuffer

class QNetwork(Model):

	def __init__(self, states, actions):
		
		last_init = tf.random_normal_initializer(stddev=0.0005)

		inputs = tf.keras.layers.Input(
			shape=(states,),
			dtype=tf.float32)
		out = tf.keras.layers.Dense(
			600,
			activation=tf.nn.leaky_relu,
			kernel_initializer=KERNEL_INITIALIZER)(inputs)
		out = tf.keras.layers.Dense(
			300,
			activation=tf.nn.leaky_relu,
			kernel_initializer=KERNEL_INITIALIZER)(out)
		outputs = tf.keras.layers.Dense(
			actions,
			activation="tanh",
			kernel_initializer=last_init)(out)

		super().__init__(inputs, outputs)

class DQL:

	def __init__(
			self, num_states, num_actions,
			gamma=GAMMA, rho=RHO, std_dev=STD_DEV):
		
		# initialize Actor and Critic networks (Main)
		self.dqn = QNetwork(
			num_states,
			num_actions)
		self.dqn_target = QNetwork(
			num_states,
			num_actions)

		weights = self.dqn.get_weights()
		self.dqn_target.set_weights(weights)

		self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
		self.gamma = tf.constant(gamma)
		self.rho = rho
		self.num_states = num_states
		self.num_actions = num_actions
		self.noise = OUActionNoise(
			mean=np.zeros(1),
			std_deviation=float(std_dev) * np.ones(1))

		# optimizers
		self.q_optimizer = tf.keras.optimizers.Adam(Q_LR, amsgrad=True)

		# temporary variable for side effects
		self.cur_action = None

		# define update weights
		@tf.function(input_signature=[
			tf.TensorSpec(shape=(None, self.num_states), dtype=tf.float32),
			tf.TensorSpec(shape=(None, ), dtype=tf.int64),
			tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
			tf.TensorSpec(shape=(None, self.num_states), dtype=tf.float32),
			tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
		])
		def update_weights(s, a, r, sn, d):
			test = [s, a, r, sn, d]
			# Update Q-Network
			with tf.GradientTape() as tape:
				tape.watch(self.dqn.trainable_variables)
				
				# Compute y
				q_values = self.dqn_target(sn)
				an = tf.argmax(q_values, axis=1)
				
				one_hot = tf.one_hot(an, self.num_actions)
				q_value = one_hot * q_values
				q_value = tf.reduce_sum(q_value, axis = 1)
				y = r + (1 - d) * self.gamma * q_value

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
			
			return q_loss
		
		self.update_weights = update_weights

	def act(self, state, _notrandom=True):
		state = tf.convert_to_tensor([state], dtype=tf.float32)
		q_value = self.dqn(state)[0].numpy()
		if _notrandom:
			self.cur_action = np.argmax(q_value)
		else:
			self.cur_action = np.random.choice(self.num_actions)
		return self.cur_action, q_value

	def remember(self, prev_state, reward, state, done):
		# record it in the buffer based on its reward
		self.buffer.append(prev_state, self.cur_action, reward, state, done)

	def can_train(self):
		memory = len(self.buffer.buffer)
		if memory > BATCH_SIZE:
			return True
		else:
			return False

	def learn(self, entry):
		s,a,r,sn,d = zip(*entry)

		q_l = self.update_weights(	
			tf.convert_to_tensor(s,dtype=tf.float32),
			tf.convert_to_tensor(a,dtype=tf.int64),
			tf.convert_to_tensor(r,dtype=tf.float32),
			tf.convert_to_tensor(sn,dtype=tf.float32),
			tf.convert_to_tensor(d,dtype=tf.float32)
		)

		self.update_target(self.dqn_target, self.dqn, self.rho)

		return q_l

	def save_weights(self, path):
		parent_dir = os.path.dirname(path)
		if not os.path.exists(parent_dir):
			os.makedirs(parent_dir)
		# Save the weights
		self.dqn.save_weights(path + "an.h5")
		self.dqn_target.save_weights(path + "at.h5")
		return "Weights Saved"

	def load_weights(self, path):
		try:
			self.dqn.load_weights(path + "an.h5")
			self.dqn_target.load_weights(path + "cn.h5")
		except:
			return "Weights cannot be loaded"
		return "Weights loaded"

	def update_target(self, target, ref, rho=0):
		weights = []
		old_weights = list(zip(target.get_weights(), ref.get_weights()))
		for (target_weight, ref_weight) in old_weights:
			w = rho * ref_weight + (1 - rho) * target_weight
			weights.append(w)

		target.set_weights(weights)
