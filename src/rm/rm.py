from rm.rm_reward import *
from rm.rm_cost import *

class SafetyMachine:

	def __init__(self, rm_file, cm_file):
		self.rm = RewardMachine(rm_file)
		self.cm = CostMachine(cm_file)

	def get_rm_states(self)
		return self.rm.U
	
	def get_cm_states(self)
		return self.cm.U

	def reset(self):
		return (
			self.rm.rest(),
			self.cm.reset()
		)


class RewardMachine:

	def __init__(self, file): # <U, u0, T, delta_u, delta_r>

		self.U = []			# list of RM states
		self.u0 = None 		# RM initial state
		self.delta_u = {}	# state-transition function
		self.delta_r = {}	# reward-transition function
		self.T = set()		# terminal state/s
		self.known_transitions = {}
	
		self._build_reward_machine(file)

	def step(self, u1, true_props, s_info={}):
		u2 = self._update_state(u1, true_props)
		done = (u2 == -1 or u2 in self.T)
		rew = self._get_reward(u1,u2,s_info)
		return u2, rew, done 

	def get_states(self):
		return self.U

	def reset(self):
		return self.u0

	# Private Methods --------------------------------------------------------
	
	def _build_reward_machine(self, file):
		f = open(file)
		lines = [l.rstrip() for l in f]
		f.close()

		self.u0 = eval(lines[0])
		
		for line in lines[1:]:
			u1, u2, dnf, reward = eval(line)
			self._add_state(u1)
			self._add_state(u2)
			self._add_transition(u1, u2, dnf, reward)
		
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

	def _add_transition(self, u1, u2, dnf, reward):
		if u1 not in self.delta_u:
			self.delta_u[u1] = {}
		self.delta_u[u1][u2] = dnf

		if u1 not in self.delta_r:
			self.delta_r[u1] = {}
		self.delta_r[u1][u2] = reward

	def _is_terminal(self,u):
		if u not in self.delta_u:
			return True
		if len(self.delta_u[u]) == 0:
			return True
		u2 = list(self.delta_u[u].keys())[0]
		if len(self.delta_u[u]) == 1 and self.delta_u[u][u2] == 'True':
			return True
		return False

	def _get_reward(self,u1,u2,s_info):
		"""
		Returns the reward associated to this transition.
		"""
		# Getting reward from the RM
		reward = 0
		if (u1 in self.delta_r and
			u2 in self.delta_r[u1]):
			reward += self.delta_r[u1][u2].get_reward(s_info)
		return reward

	def _update_state(self, u1, true_props):
		if (u1, true_props) not in self.known_transitions:
			u2 = -1
			for u in self.delta_u[u1]:
				trans = self.delta_u[u1][u]
				if self._evaluate_prop(trans, true_props):
					u2 = u
			self.known_transitions[(u1, true_props)] = u2
		else:
			u2 = self.known_transitions[(u1, true_props)]
		return u2

	def _evaluate_prop(self, trans, true_props):

		# ORs
		if "|" in trans:
			for f in trans.split("|"):
				if self._evaluate_prop(f,true_props):
					return True
			return False
		# ANDs
		if "&" in trans:
			for f in trans.split("&"):
				if not self._evaluate_prop(f,true_props):
					return False
			return True
		# NOT
		if trans.startswith("!"):
			return not self._evaluate_prop(trans[1:],true_props)

		# Base cases
		if trans == "True":  return True
		if trans == "False": return False
		return trans in true_props

class CostMachine:

	def __init__(self, file): # <U, u0, T, delta_u, delta_d>

		self.U = []			# list of RM states
		self.u0 = None 		# RM initial state
		self.delta_u = {}	# state-transition function
		self.delta_d = {}	# cost-transition function
		self.T = set()		# terminal state/s
		self.known_transitions = {}
	
		self._build_cost_machine(file)

	def step(self, u1, rm_u, true_props, s_info={}):
		u2 = self._update_state(u1, rm_u, true_props)
		done = (u2 == -1 or u2 in self.T)
		d = self._get_cost(u1,u2,s_info)
		return u2, d, done

	def get_states(self):
		return self.U

	def reset(self):
		return self.u0

	# Private Methods --------------------------------------------------------

	def _build_cost_machine(self, file):
		f = open(file)
		lines = [l.rstrip() for l in f]
		f.close()

		self.u0 = eval(lines[0])
		
		for line in lines[1:]:
			u1, u2, rm_u, dnf, cost = eval(line)
			self._add_state(u1)
			self._add_state(u2)
			self._add_transition(u1, u2, rm_u, dnf, cost)
		
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

	def _add_transition(self, u1, u2, rm_u, dnf, cost):
		if u1 not in self.delta_u:
			self.delta_u[u1] = {}
		self.delta_u[u1][u2] = [rm_u, dnf]

		if u1 not in self.delta_d:
			self.delta_d[u1] = {}
		self.delta_d[u1][u2] = cost

	def _is_terminal(self,u):
		if u not in self.delta_u:
			return True
		if len(self.delta_u[u]) == 0:
			return True
		u2 = list(self.delta_u[u].keys())[0]
		if (len(self.delta_u[u]) == 1 and 
			self.delta_u[u][u2][1] == 'True'):
			return True
		return False

	def _update_state(self, u1, rm_u, true_props):
		if (u1, true_props) not in self.known_transitions:
			u2 = -1
			for u in self.delta_u[u1]:
				print(u)
				trans_u = self.delta_u[u1][u][0]
				trans = self.delta_u[u1][u][1]
				valid_rm_trans = (
					trans_u == rm_u or
					trans_u == None
				)
				valid_trans = self._evaluate_prop(trans, true_props)
				print(valid_rm_trans)
				print(valid_trans)
				if (valid_rm_trans and
					valid_trans):
					u2 = u
			self.known_transitions[(u1, true_props)] = u2
		else:
			u2 = self.known_transitions[(u1, true_props)]
		return u2

	def _evaluate_prop(self, trans, true_props):

		# ORs
		if "|" in trans:
			for f in trans.split("|"):
				if self._evaluate_prop(f,true_props):
					return True
			return False
		# ANDs
		if "&" in trans:
			for f in trans.split("&"):
				if not self._evaluate_prop(f,true_props):
					return False
			return True
		# NOT
		if trans.startswith("!"):
			return not self._evaluate_prop(trans[1:],true_props)

		# Base cases
		if trans == "True":  return True
		if trans == "False": return False
		return trans in true_props

	def _get_cost(self,u1,u2,s_info):
		cost = 0
		if (u1 in self.delta_d and
			u2 in self.delta_d[u1]):
			cost += self.delta_d[u1][u2].get_cost(s_info)
		return cost

if __name__ == "__main__":
	# execute only if run as a script
	files = [
		'../envs/water/rm/rm.txt',
		'../envs/water/rm/srm.txt'
	]
	rm = RewardMachine(files[0])
	srm = CostMachine(files[1])

	new_u, cost, done = srm.step(0,2,'e')
	print('Transition to: {}'.format(new_u))
	print('Cost of transition: {}'.format(cost))
	if done:
		print('Done')
	else:
		print('Not Done')