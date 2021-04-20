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
	
	def _build_reward_machine(self, file):
		f = open(file)
		lines = [l.rstrip() for l in f]
		f.close()

		self.u0 = eval(lines[0])
		
		for line in lines[1:]:
			u1, u2, dnf, reward, constraint= eval(line)
			self._add_state(u1)
			self._add_state(u2)
			self._add_transition(u1, u2, dnf, reward, constraint)
		
		for u in self.U:
			if self._is_terminal(u):
				self.T.add(u)

	def _add_state(self, u):
		if u not in self.U: 
			self.U.append(u)

	def _add_states(self, l_states):
		for u in l_states:
			if u not in self.U: 
				self.U.append(u)

	def _add_transition(self, u1, u2, dnf, reward, constraint):
		if u1 not in self.delta_u:
			self.delta_u[u1] = {}
		self.delta_u[u1][u2] = dnf

		if u1 not in self.delta_r:
			self.delta_r[u1] = {}
		self.delta_r[u1][u2] = reward

		if u1 not in self.delta_d:
			self.delta_d[u1] = {}
		self.delta_d[u1][u2] = constraint

	def _is_terminal(self,u):
		if u not in self.delta_u:
			return True
		if len(self.delta_u[u]) == 0:
			return True
		u2 = list(self.delta_u[u].keys())[0]
		if len(self.delta_u[u]) == 1 and self.delta_u[u][u2] == 'True':
			return True
		return False

if __name__ == '__main__':

	rm = RewardMachine('rm.txt')
	rm.print_rm()

