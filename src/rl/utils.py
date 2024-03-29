import datetime
import random

import numpy as np
import tensorflow as tf

from collections import deque


class OUActionNoise:
	"""
	Noise as defined in the DDPG algorithm
	"""

	def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
		self.theta = theta
		self.mean = mean
		self.std_dev = std_deviation
		self.dt = dt
		self.x_initial = x_initial
		self.reset()

	def __call__(self):
		# Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
		x = (
				self.x_prev
				+ self.theta * (self.mean - self.x_prev) * self.dt
				+ self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
		)
		# Store x into x_prev
		# Makes next noise dependent on current one
		self.x_prev = x
		return x

	def reset(self):
		if self.x_initial is not None:
			self.x_prev = self.x_initial
		else:
			self.x_prev = np.zeros_like(self.mean)


# buffer params
UNBALANCE_P = 0.8
BUFFER_UNBALANCE_GAP = 0.5


class ReplayBuffer:
	"""
	Replay Buffer to store the experiences.
	"""

	def __init__(self, buffer_size, batch_size):
		"""
		Initialize the attributes.
		Args:
			buffer_size: The size of the buffer memory
			batch_size: The batch for each of the data request `get_batch`
		"""
		self.buffer = deque(maxlen=int(buffer_size))  # with format of (s,a,r,s')

		# constant sizes to use
		self.batch_size = batch_size

		# temp variables
		self.p_indices = [BUFFER_UNBALANCE_GAP/2]

	def append(self, state, action, r, d, sn, done):
		"""
		Append to the Buffer
		Args:
			state: the state
			action: the action
			r: the reward
			d: the cost
			sn: the next state
			done: whether one loop is done or not
		"""
		self.buffer.append([state, action, np.expand_dims(r, -1), np.expand_dims(d, -1), sn, np.expand_dims(done, -1)])

	def get_batch(self, unbalance_p=True):
		"""
		Get the batch randomly from the buffer
		Args:
			unbalance_p: If true, unbalance probability of taking the batch from buffer with
			recent event being more prioritized
		Returns:
			the resulting batch
		"""
		# unbalance indices
		p_indices = None
		if random.random() < unbalance_p:
			self.p_indices.extend((np.arange(len(self.buffer)-len(self.p_indices))+1)
								  * BUFFER_UNBALANCE_GAP + self.p_indices[-1])
			p_indices = self.p_indices / np.sum(self.p_indices)

		chosen_indices = np.random.choice(len(self.buffer),
										  size=min(self.batch_size, len(self.buffer)),
										  replace=False,
										  p=p_indices)

		buffer = [self.buffer[chosen_index] for chosen_index in chosen_indices]

		return buffer
