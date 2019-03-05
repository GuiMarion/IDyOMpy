from idyom import data

class markovChain():
	"""
	Module that define MarkovChain model and usefull functions for the project

	:param order: order of the Markov Chain
	:param alphabetSize: number of elements in the alphabet

	:type order: int
	:type alphabetSize: int
	"""
	def __init__(self, order, alphabetSize):

		# order of the model
		self.order = order

		# matrix contening the transition probabilities
		self.matrix = np.zeros((alphabetSize, alphabetSize))

	def train(self, data):
		"""
		Fill the matrix from data
		
		:param data: pre-processed data to train with
		:type data: data object
		"""
		return 0

	def getPrediction(self, state):
		"""
		Return the probability distribution from a given state
		
		:param state: a sequence of viewPoints of sier order
		:type state: np.array(order)

		:return: np.array(alphabetSize).astype(float)
	
		"""

		# return a row in the matrix
		return 0

	def getLikelihood(self, state, note):
		"""
		Return the likelihood of a note given a state
		
		:param state: a sequence of viewPoints of sier order
		:param note: integer or name of the note

		:type state: np.array(order)
		:type note: int or string

		:return: float value of the likelihood
		"""
		return 0

	def save(self, file):
		"""
		Save a trained model
		
		:param file: path to the file
		:type file: string
		"""

		return 0

	def load(self, path):
		"""
		Load a trained model

		:param path: path to the file
		:type path: string
		"""

		return 0
