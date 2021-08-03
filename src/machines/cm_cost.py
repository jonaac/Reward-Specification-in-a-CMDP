import math

class CostFunction:
	def __init__(self):
		pass

	# To implement...
	def get_cost(self, s_info):
		raise NotImplementedError("To be implemented")

	def get_type(self):
		raise NotImplementedError("To be implemented")


class ConstantCost(CostFunction):

	def __init__(self, c):
		super().__init__()
		self.c = c

	def get_type(self):
		return "constant"

	def get_cost(self, s_info):
		return self.c