from idyom import data
from idyom import markovChain

class longTermModel():
	def __init__(self, maxorder, alphabetSize=100):

		# Maximal order of the model
		self.maxOrder = maxOrder

		# list contening different-order markov chains
		self.models = []
		for order in range(maxOrder):
			models.append(markovChain.markovChain(order, alphabetSize))

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
