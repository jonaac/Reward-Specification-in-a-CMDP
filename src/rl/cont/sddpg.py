import os
import sys
from collections import deque

import osqp
import numpy as np
from scipy import sparse
import tensorflow as tf
from tensorflow.keras import Model

from rl.parameters import (KERNEL_INITIALIZER, GAMMA, RHO, STD_DEV, BUFFER_SIZE,
							BATCH_SIZE, Q_LR, NUM_STEPS, COST_BOUND)
from rl.utils import OUActionNoise, ReplayBuffer
from rl.cont.ddpg import ActorNetwork, CriticNetwork, DDPG


class CostNetwork(Model):
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


class SDDPG(DDPG):

	def __init__(
			self, num_states, num_actions, action_high,
			gamma=GAMMA, rho=RHO, std_dev=STD_DEV):
		
		super().__init__(
			num_states, num_actions, action_high,
			GAMMA, RHO, STD_DEV)

		self.cost_network = CostNetwork(num_states, num_actions, action_high)
		self.cost_target = CostNetwork(num_states, num_actions, action_high)

		self.cost_target.set_weights(self.cost_network.get_weights())

		self.cost_optimizer = tf.keras.optimizers.Adam(COST_LR,amsgrad=True)

		self.baseline_policy = ActorNetwork(num_states, num_actions, action_high)
		self.baseline_policy.set_weights(self.actor_network.get_weights())
		self.weight_stack = deque()

		# define update weights
		@tf.function(input_signature=[
			tf.TensorSpec(shape=(None, self.num_states), dtype=tf.float32),
			tf.TensorSpec(shape=(None, self.num_actions), dtype=tf.float32),
			tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
			tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
			tf.TensorSpec(shape=(None, self.num_states), dtype=tf.float32),
			tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
		])
		def update_weights(s, a, r, d, sn, done):
			# ---------------------------------------------------------------------------- #
			# Update Critic network
			with tf.GradientTape() as tape:
				tape.watch(self.critic_network.trainable_variables)
				
				# Compute Target y
				target = self.critic_target([sn, self.actor_target(sn)])
				y = r + self.gamma * (1-done) * target
				
				# Compute Delta Q
				error = tf.math.abs(y - self.critic_network([s, a]))
				critic_loss = tf.math.reduce_mean(error)

			critic_grad = tape.gradient(
				critic_loss,
				self.critic_network.trainable_variables)
			self.critic_optimizer.apply_gradients(
				zip(critic_grad, self.critic_network.trainable_variables))
			# ---------------------------------------------------------------------------- #
			# Update Cost Network
			with tf.GradientTape() as tape:
				tape.watch(self.cost_network.trainable_variables)
				
				# Compute Target y
				target = self.cost_target([sn, self.actor_target(sn)])
				z = d + self.gamma * (1-done) * target
				
				# Compute Delta Q
				error = tf.math.abs(z - self.cost_network([s, a]))
				cost = tf.math.reduce_mean(error)

			cost_grad = tape.gradient(
				cost,
				self.cost_network.trainable_variables)
			self.cost_optimizer.apply_gradients(
				zip(cost_grad, self.cost_network.trainable_variables))
			# ---------------------------------------------------------------------------- #
			# Update Actor network
			with tf.GradientTape() as tape:
				tape.watch(self.actor_network.trainable_variables)
				# define the delta mu
				target = self.critic_network([s, self.actor_network(s)])
				actor_loss = -tf.math.reduce_mean(target)
			
			actor_grad = tape.gradient(
				actor_loss,
				self.actor_network.trainable_variables)
			self.actor_optimizer.apply_gradients(
				zip(actor_grad, self.actor_network.trainable_variables))

			if len(self.weight_stack) == NUM_STEPS:
				self.baseline_policy.set_weights(self.weight_stack.popleft())
				self.weight_stack.append(self.actor_network.get_weights())
			else:
				self.weight_stack.append(self.actor_network.get_weights())
			
			return critic_loss, actor_loss, cost
	
	# Update ACT function
	def act(self, state, _notrandom=True, noise=True):
		_random = np.random.uniform(self.action_low,
									self.action_high,
									self.num_actions)
		_noise = (self.noise() if noise else 0)
		if _notrandom:
			state = tf.convert_to_tensor([state], dtype=tf.float32)
			self.cur_action = self.actor_network(state).numpy()
		else:
			self.cur_action = _random
			return self.cur_action
		
		self.cur_action = np.clip(
			self.cur_action,
			self.action_low, 
			self.action_high)

		# ------ Solve QP Problem ----------- #
		I = sparse.identity(self.num_actions, format="csc")  # [n,n]

		cost = self.cost_network([state, self.actor_network(state)])
		e = ((1 - self.gamma) * (COST_BOUND - cost)).numpy()[0]  # [1,]

		action = self.baseline_policy(state)
		with tf.GradientTape() as tape:
			tape.watch(action)
			lyapunov_cost = self.cost_network([state, action])
		g = tape.gradient(
				lyapunov_cost,
				action).numpy() # [1,n]

		cur_action = self.cur_action.T  # [n,1]

		P = I  # [n,n]
		q = (- I @ cur_action).T[0]  # [n,] = - [n,n] @ [n,1]

		A = g  # [1,n]
		A = sparse.csc_matrix(A)
		u = e + (cur_action.T @ g.T)[0]  # [1,] = [1,1] + ([1,n] @ [n,1])
		l = np.array([- np.inf])

		prob = osqp.OSQP()
		# Setup workspace and change alpha parameter
		prob.setup(P, q, A, l, u, alpha=1.0, verbose=0)
		# Solve problem
		results = prob.solve()

		self.cur_action = results.x

		self.cur_action = np.clip(
			self.cur_action,
			self.action_low, 
			self.action_high)
		
		return self.cur_action
	
	def learn(self, entry):
		s, a, r, _, sn, done = zip(*entry)

		c_l, a_l, cost = self.update_weights(	
			tf.convert_to_tensor(s, dtype=tf.float32),
			tf.convert_to_tensor(a, dtype=tf.float32),
			tf.convert_to_tensor(r, dtype=tf.float32),
			tf.convert_to_tensor(sn, dtype=tf.float32),
			tf.convert_to_tensor(done, dtype=tf.float32)
		)

		self.update_target(self.actor_target, self.actor_network, self.rho)
		self.update_target(self.critic_target, self.critic_network, self.rho)
		self.update_target(self.cost_target, self.critic_network, self.rho)

		return c_l, a_l, cost

	def save_weights(self, path):
		parent_dir = os.path.dirname(path)
		if not os.path.exists(parent_dir):
			os.makedirs(parent_dir)
		# Save the weights
		self.actor_network.save_weights(path + "an.h5")
		self.critic_network.save_weights(path + "cn.h5")
		self.cost_network.save_weights(path + "co_n.h5")
		
		self.critic_target.save_weights(path + "ct.h5")
		self.actor_target.save_weights(path + "at.h5")
		self.cost_target.save_weights(path + "co_t.h5")

	def load_weights(self, path):
		try:
			self.actor_network.load_weights(path + "an.h5")
			self.critic_network.load_weights(path + "cn.h5")
			self.cost_network.load_weights(path + "co_n.h5")
			
			self.critic_target.load_weights(path + "ct.h5")
			self.actor_target.load_weights(path + "at.h5")
			self.cost_target.load_weights(path + "co_t.h5")
		except:
			return "Weights cannot be loaded"
		return "Weights loaded"


