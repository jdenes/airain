


class Trader(object):

	def __init__(self, name, agent_type, freq, h, seed, forcast):
		self.name = name
		self.agent_type = agent_type
		self.freq = freq
		self.h = h
		self.forcast = 1
		self.seed = seed

	def prepare_data(self, csv_file):
	'''
	Converts .csv file to appropriate dataformat, depending on agent type.
	'''
		self.X_train = 


	def train(self):
	'''
	Using prepared data, trains model depending on agent type.
	'''
		pass