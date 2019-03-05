from idyom import data
from idyom import markovChain

class longTermModel():
	"""
	Module implementing the Long Term Model from IDyOM, thid model contains several Markov Chains of different orders averaged with the weights of their shanon entropy.

	:param viewPoint: viewPoint to use, cf. data.getViewPoints()
	:param maxOrder: maximal order of the models
	:param alphabetSize(optional): size of the alphabet, number of viewPoints value to take account in

	:type viewPoint: string
	:type maxOrder: int
	:type alphabetSize(optional): int
	"""
	def __init__(self, viewPoint, maxOrder, alphabetSize=100):

		# Maximal order of the model
		self.maxOrder = maxOrder

		# ViewPoint to use
		self.viewPoint = viewPoint

		# list contening different-order markov chains
		self.models = []
		for order in range(maxOrder):
			models.append(markovChain.markovChain(order, alphabetSize))

	def train(self, data):
		""" 
		Fill the matrix from data
		
		:param data: data to train from

		:type data: class data
		"""
		return 0

	def getPrediction(sequence):
		"""
		Returns the probability distribution from a given state
		
		:param sequence: a sequence of viewPoint data, cf. data.getData(viewPoint)

		:type sequence: np.array(length)

		:return: dictionary, dico[note] = likelihood (float)
		"""

		# return a row in the matrix
		return 0

	def getLikelihood(self, state, note):
		"""
		Returns the likelihood of a note given a state
		
		:param state: a sequence of viewPoint data, cf. data.getData(viewPoint)
		:param note: the interger or name of the note

		:type state: np.array(length)
		:type note:	int or string

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
