import numpy as np 

class RewardMachine:

	def __init__(self, file): # <U, u0, T, delta_u, delta_r, delta_d>

		self.U = []			# list of RM states
		self.u0 = None 		# RM initial state
		self.delta_u = {}	# state-transition function
		self.delta_r = {}	# reward-transition function
		self.delta_d = {}	# constraint-transition function
		self.T = set()		# terminal state
	
		self._build_reward_machine(file)


	# Private Methods --------------------------------------------------------
	
	def _build_reward_machine(file):


if __name__ == '__main__':

	f = open(rm.txt)
	lines = [l.rstrip() for l in f]
	f.close()
