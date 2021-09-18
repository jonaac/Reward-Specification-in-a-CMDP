import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from rl.parameters import (KERNEL_INITIALIZER, GAMMA, RHO, STD_DEV, 
							BUFFER_SIZE, BATCH_SIZE, Q_LR)
from rl.utils import OUActionNoise, ReplayBuffer
from rl.disc.dqn import QNetwork, DQL


class DQRM(DQL):

	def __init__(
			self, num_states, num_actions,
			gamma=GAMMA, rho=RHO, std_dev=STD_DEV):
		
		super().__init__(num_states, num_actions, GAMMA, RHO, STD_DEV)

	def remember(self, experiences):
		# record it in the buffer based on its reward
		for experience in experiences:
			prev_state, action, rm_r, cm_d, state, done = zip(*experience)
			self.buffer.append(prev_state, action, rm_r, cm_d, state, done)
