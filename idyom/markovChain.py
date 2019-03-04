import data

class markovChain():
	def __init__(self, order, alphabetSize):

		# order of the model
		self.order = order

		# matrix contening the transition probabilities
		self.matrix = np.zeros((alphabetSize, alphabetSize))

	def train(self, data):
		# fill the matrix from data

		return 0

	def getPrediction(state):
		# return the probability distribution from a given state

		# return a row in the matrix
		return 0

	def getLikelihood(self, state, note):
		# return the likelihood of a note given a state

		return 0

	def save(self, file):
		# save a trained model

		return 0

	def load(self, path):
		# load a trained model

		return 0
