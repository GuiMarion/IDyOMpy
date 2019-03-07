from idyom import data

import numpy as np
import pickle
#from tqdm import tqdm

# We store state transition for now, mostly for debug reasons
# at some point, we will be able to only store state to notes transitions
# this can faster the training part and the storage

class markovChain():
	"""
	Module that define MarkovChain model and usefull functions for the project

	:param order: order of the Markov Chain (>=1)
	:param alphabetSize: number of elements in the alphabet

	:type order: int
	:type alphabetSize: int
	"""
	def __init__(self, order, alphabetSize=None):

		# order of the model
		self.order = order

		# dictionary containing the transition probabilities between states
		self.transitions = {}

		# dictionary containing containing the probabilities bewteen states and notes
		self.probabilities = {}

		# alphabet of state of the data
		self.stateAlphabet = []

		# alphabet of notes of the data
		self.alphabet = []

		if order < 1:
			raise(ValueError("order should be at least grater than 1."))

	def train(self, data):
		"""
		Fill the matrix from data, len(data) should be greater than the order.
		
		:param data: pre-processed data to train with
		:type data: data object
		"""

		if len(data) <= self.order:
			raise(ValueError("We cannot train a model with less data than the order of the model."))


		SUM = {}

		# iterating over data
		for i in range(len(data) - self.order*2 - 1):
			state = str(list(data[i:i+self.order]))
			# constructing alphabet
			if state not in self.transitions:
				self.stateAlphabet.append(state)
				SUM[state] = 0
				self.transitions[state] = {}
				self.probabilities[state] = {}

			target = str(list(data[i+self.order:i+self.order*2]))
			target_elem = str(list(data[i+self.order:i+self.order*2])[0])

			if target_elem not in self.alphabet:
				self.alphabet.append(target_elem)

			# constructing state transitions
			if target not in self.transitions[state]:
				self.transitions[state][target] = 1
			else:
				self.transitions[state][target] += 1

			# constructing state to note transitions
			if target_elem not in self.probabilities[state]:
				self.probabilities[state][target_elem] = 1
			else:
				self.probabilities[state][target_elem] += 1

			SUM[state] += 1

		# We devide by the number of occurence for each state
		for state in self.transitions:
			for target in self.transitions[state]:
				self.transitions[state][target] /= SUM[state]

		for state in self.probabilities:
			for target in self.probabilities[state]:
				self.probabilities[state][target] /= SUM[state]

		# We sort the alphabet
		self.alphabet.sort()
		self.stateAlphabet.sort()

	def getPrediction(self, state):
		"""
		Return the probability distribution of notes from a given state
		
		:param state: a sequence of viewPoints of sier order
		:type state: np.array(order)

		:return: np.array(alphabetSize).astype(float)
	
		"""

		# return a row in the matrix

		if state in self.probabilities:
			return self.probabilities[state]
		else:
			print("We never saw this state in database")
			return None

	def getLikelihood(self, state, note):
		"""
		Return the likelihood of a note given a state
		
		:param state: a sequence of viewPoints of sier order
		:param note: integer or name of the note

		:type state: np.array(order)
		:type note: int or string

		:return: float value of the likelihood
		"""

		# in order to work with numpyn array and list
		if not isinstance(state, str):
			state = str(list(state))

		if state in self.probabilities:
			pass
		else:
			print("We never saw this state in database.")
			return 0

		if str(note) in self.probabilities[state]:
			return self.probabilities[state][str(note)]
		else:
			print("We never saw this transition in database.")
			return 0


	def save(self, file):
		"""
		Save a trained model
		
		:param file: path to the file
		:type file: string
		"""

		f = open(file, 'wb')
		pickle.dump(self.__dict__, f, 2)
		f.close()

	def load(self, path):
		"""
		Load a trained model

		:param path: path to the file
		:type path: string
		"""

		f = open(path, 'rb')
		tmp_dict = pickle.load(f)
		f.close()          

		self.__dict__.update(tmp_dict) 


	def getStatesMatrix(self):
		"""
		Return the transition matrix between states made from the dictionnary

		:return: transition matrix (np.array())
		"""



		matrix = np.zeros((len(self.stateAlphabet), len(self.stateAlphabet)))
		k1 = 0
		k2 = 0

		for state in self.stateAlphabet:
			k2 = 0
			for target in self.stateAlphabet:
				if state in self.transitions and target in self.transitions[state]:
					matrix[k1][k2] = self.transitions[state][target]
				else:
					matrix[k1][k2] = 0.0
				k2 += 1
			k1 += 1

		return matrix


	def getMatrix(self):
		"""
		Return the transition matrix between states and notes

		:return: transition matrix (np.array())
		"""



		matrix = np.zeros((len(self.stateAlphabet), len(self.alphabet)))
		k1 = 0
		k2 = 0

		for state in self.stateAlphabet:
			k2 = 0
			for target in self.alphabet:
				if state in self.transitions and target in self.probabilities[state]:
					matrix[k1][k2] = self.probabilities[state][target]
				else:
					matrix[k1][k2] = 0
				k2 += 1
			k1 += 1

		return matrix


	def generate(length):
		"""
		Implement a very easy random walk in order to generate a sequence

		:param length: length of the generated sequence
		:type length: int

		:return: sequence (np.array()) 
		"""

		pass