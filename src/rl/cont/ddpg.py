import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from rl.parameters import (KERNEL_INITIALIZER, GAMMA, RHO, STD_DEV, 
								BUFFER_SIZE, BATCH_SIZE, CRITIC_LR, ACTOR_LR)
from rl.utils import OUActionNoise, ReplayBuffer


class ActorNetwork(Model):

	def __init__(self, states, actions, action_max):

		super(ActorNetwork, self).__init__()

		self.action_max = action_max

		last_init = tf.random_normal_initializer(stddev=0.0005)

		self.inputs = tf.keras.layers.Dense(states)
		
		self.layer_1 = tf.keras.layers.Dense(
			600,
			activation=tf.nn.leaky_relu,
			kernel_initializer=KERNEL_INITIALIZER
		)
		
		self.layer_2 = tf.keras.layers.Dense(
			300,
			activation=tf.nn.leaky_relu,
			kernel_initializer=KERNEL_INITIALIZER
		)
		
		self.outputs = tf.keras.layers.Dense(
			actions,
			activation="tanh",
			kernel_initializer=last_init
		)
	
	def call(self, entry):
		inputs = self.inputs(entry)
		layer_1 = self.layer_1(inputs)
		layer_2 = self.layer_2(layer_1)
		return self.outputs(layer_2) * self.action_max

class CriticNetwork(Model):

	def __init__(self, states, actions, action_max):
		
		last_init = tf.random_normal_initializer(stddev=0.00005)

		# State input
		state_input = tf.keras.layers.Input(shape=(states,), dtype=tf.float32)
		state_out = tf.keras.layers.Dense(
			600, activation=tf.nn.leaky_relu,
			kernel_initializer=KERNEL_INITIALIZER)(state_input)
		state_out = tf.keras.layers.BatchNormalization()(state_out)
		state_out = tf.keras.layers.Dense(
			300, activation=tf.nn.leaky_relu,
			kernel_initializer=KERNEL_INITIALIZER)(state_out)

		# Action input
		action_input = tf.keras.layers.Input(shape=(actions), dtype=tf.float32)
		action_out = tf.keras.layers.Dense(
			300, activation=tf.nn.leaky_relu,
			kernel_initializer=KERNEL_INITIALIZER)(action_input/action_max)

		# Concatenate Layers
		added = tf.keras.layers.Add()([state_out, action_out])

		added = tf.keras.layers.BatchNormalization()(added)
		outs = tf.keras.layers.Dense(
			150, activation=tf.nn.leaky_relu,
			kernel_initializer=KERNEL_INITIALIZER)(added)
		outs = tf.keras.layers.BatchNormalization()(outs)
		# outs = tf.keras.layers.Dropout(DROUPUT_N)(outs)
		outputs = tf.keras.layers.Dense(1, kernel_initializer=last_init)(outs)

		# Outputs single value for given state-action
		super().__init__([state_input, action_input], outputs)

def update_target(target, ref, rho=0):
	weights = []
	old_weights = list(zip(target.get_weights(), ref.get_weights()))
	for (target_weight, ref_weight) in old_weights:
		w = rho * ref_weight + (1 - rho) * target_weight
		weights.append(w)
	
	target.set_weights(weights)

class DDPG:

	"""
	Need to import: GAMMA, RHO, STD_DEV
	"""
	def __init__(self, num_states, num_actions, action_high, 
				action_low, gamma=GAMMA, rho=RHO, std_dev=STD_DEV):
		
		# initialize Actor and Critic networks (Main)
		self.actor_network = ActorNetwork(num_states,  num_actions, action_high)
		self.critic_network = CriticNetwork(num_states,
											num_actions,
											action_high)

		# initialize Actor and Critic networks (Target)
		self.actor_target = ActorNetwork(num_states, num_actions, action_high)
		self.critic_target = CriticNetwork(num_states, num_actions, action_high)

		# setting the same weights for Main and Target networks
		self.actor_target.set_weights(self.actor_network.get_weights())
		self.critic_target.set_weights(self.critic_network.get_weights())

		self.buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
		self.gamma = tf.constant(gamma)
		self.rho = rho
		self.action_high = action_high
		self.action_low = action_low
		self.num_states = num_states
		self.num_actions = num_actions
		self.noise = OUActionNoise(
			mean=np.zeros(1),
			std_deviation=float(std_dev) * np.ones(1))

		# optimizers
		self.critic_optimizer = tf.keras.optimizers.Adam(CRITIC_LR,amsgrad=True)
		self.actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR,amsgrad=True)

		# temporary variable for side effects
		self.cur_action = None

		# define update weights
		@tf.function(input_signature=[
			tf.TensorSpec(shape=(None, self.num_states), dtype=tf.float32),
			tf.TensorSpec(shape=(None, self.num_actions), dtype=tf.float32),
			tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
			tf.TensorSpec(shape=(None, self.num_states), dtype=tf.float32),
			tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
		])
		def update_weights(s, a, r, sn, d):
			# Update Critic network
			with tf.GradientTape() as tape:
				tape.watch(self.critic_network.trainable_variables)
				
				# Compute Target y
				target = self.critic_target([sn, self.actor_target(sn)])
				y = r + self.gamma * (1-d) * target
				
				# Compute Delta Q
				error = tf.math.abs(y - self.critic_network([s, a]))
				critic_loss = tf.math.reduce_mean(error)

			critic_grad = tape.gradient(critic_loss,
										self.critic_network.trainable_variables)
			self.critic_optimizer.apply_gradients(
				zip(critic_grad, self.critic_network.trainable_variables))

			# Update Actor network
			with tf.GradientTape() as tape:
				tape.watch(self.actor_network.trainable_variables)
				# define the delta mu
				target = self.critic_network([s, self.actor_network(s)])
				actor_loss = -tf.math.reduce_mean(target)
			
			actor_grad = tape.gradient(
				actor_loss,	self.actor_network.trainable_variables)
			self.actor_optimizer.apply_gradients(
				zip(actor_grad, self.actor_network.trainable_variables))
			
			return critic_loss, actor_loss
		
		self.update_weights = update_weights

	"""
	NEEDS REVIEW, SEEMS LIKE OUPUT IS UNI-DIMENSIONAL
	"""
	def act(self, state, _notrandom=True, noise=True):
		_random = np.random.uniform(self.action_low,
									self.action_high,
									self.num_actions)
		_noise = (self.noise() if noise else 0)
		if _notrandom:
			state = tf.convert_to_tensor([state], dtype=tf.float32)
			self.cur_action = self.actor_network(state)[0].numpy()
		else:
			self.cur_action = _random
		self.cur_action += _noise
		self.cur_action = np.clip(
			self.cur_action,
			self.action_low, 
			self.action_high)

		return self.cur_action

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

		c_l, a_l = self.update_weights(	
			tf.convert_to_tensor(s,dtype=tf.float32),
			tf.convert_to_tensor(a,dtype=tf.float32),
			tf.convert_to_tensor(r,dtype=tf.float32),
			tf.convert_to_tensor(sn,dtype=tf.float32),
			tf.convert_to_tensor(d,dtype=tf.float32)
		)

		update_target(self.actor_target, self.actor_network, self.rho)
		update_target(self.critic_target, self.critic_network, self.rho)

		return c_l, a_l

	def save_weights(self, path):
		parent_dir = os.path.dirname(path)
		if not os.path.exists(parent_dir):
			os.makedirs(parent_dir)
		# Save the weights
		self.actor_network.save_weights(path + "an.h5")
		self.critic_network.save_weights(path + "cn.h5")
		self.critic_target.save_weights(path + "ct.h5")
		self.actor_target.save_weights(path + "at.h5")

	def load_weights(self, path):
		try:
			self.actor_network.load_weights(path + "an.h5")
			self.critic_network.load_weights(path + "cn.h5")
			self.critic_target.load_weights(path + "ct.h5")
			self.actor_target.load_weights(path + "at.h5")
		except:
			return "Weights cannot be loaded"
		return "Weights loaded"
