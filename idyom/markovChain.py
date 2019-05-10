from idyom import data
from idyom import score

import numpy as np
import pickle
from tqdm import tqdm
import ast
import math
import warnings

THRESHOLD = 0
DEBUG = False

# We store state transition for now, mostly for debug reasons
# at some point, we will be able to only store state to notes transitions
# this can faster the training part and the storage
# And also improve efficiency and we will be able to train on more data

class markovChain():
	"""
	Module defining MarkovChain model and usefull functions for the project

	:param alphabetSize: number of elements in the alphabet
	:param VERBOSE: print some strange behoviors, for example if asking for unknwown states

	:type order: int
	:type VERBOSE: bool
	"""
	def __init__(self, order, depth=0, VERBOSE=False):

		# order of the model
		self.order = order

		# depth of the model
		self.depth = depth

		# dictionary containing the transition probabilities between states
		self.transitions = {}

		# dictionary containing containing the probabilities bewteen states and notes
		self.probabilities = {}

		# alphabet of state of the data
		self.stateAlphabet = []

		# alphabet of notes of the data
		self.alphabet = []

		self.VERBOSE = VERBOSE

		# In order to store entropy
		self.entropies = {}

		if order < 1:
			raise(ValueError("order should be at least grater than 1."))

	def train(self, dataset, reverse=False):
		"""
		Fill the matrix from data, len(data) should be greater than the order.
		
		:param data: pre-processed data to train with
		:type data: data object or list of int
		"""

		if not isinstance(dataset, list) :
			dataset = [dataset]
		self.usedScores = 0

		SUM = {}
		for data in dataset:
			if len(data) < self.order*2 + self.depth +1:
				warnings.warn("We cannot train a model with less data than the order of the model, so we skip this data.")

			else:
				self.usedScores += 1
				# iterating over data
				for i in range(len(data) - self.order*2 -self.depth +1):

					# case of the reverse prediction, we take a state get the probability to come back to the current state
					if reverse is True:
						state = str(list(data[i+self.order + self.depth : i+self.order*2 + self.depth]))
						target = str(list(data[i:i+self.order]))
						target_elem = str(list(data[i:i+self.order])[0])

					else:
						state = str(list(data[i:i+self.order]))
						target = str(list(data[i+self.order + self.depth : i+self.order*2 + self.depth]))
						target_elem = str(list(data[i+self.order + self.depth : i+self.order*2 + self.depth])[0])


					# constructing alphabet
					if state not in self.transitions:
						self.stateAlphabet.append(state)
						SUM[state] = 0
						self.transitions[state] = {}
						self.probabilities[state] = {}

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

		# We delete states that have less than THRESHOLD occurences
		for state in SUM:
			if SUM[state] < THRESHOLD:
				self.stateAlphabet.remove(state)
				if state in self.transitions:
					self.transitions.pop(state)
				if state in self.probabilities:
					self.probabilities.pop(state)

		# We devide by the number of occurence for each state
		for state in self.transitions:
			for target in self.transitions[state]:
				self.transitions[state][target] /= SUM[state]

		for state in self.probabilities:
			for target in self.probabilities[state]:
				self.probabilities[state][target] /= SUM[state]

		self.sum = SUM

		# We sort the alphabet
		self.alphabet.sort()
		self.stateAlphabet.sort()

		for state in self.stateAlphabet:
			self.getEntropy(state)

	def getPrediction(self, state):
		"""
		Return the probability distribution of notes from a given state
		
		:param state: a sequence of viewPoints such as len(state) = order
		:type state: str(np.array(order))

		:return: dictionary | dico[note] = P(note|state)
	
		"""

		# return a row in the matrix


		if not isinstance(state, str):
			state = str(list(state))

		if state in self.probabilities:
			return self.probabilities[state]
		else:
			if self.VERBOSE:
				print("We never saw this state in database")
			return None

	def getLikelihood(self, state, note):
		"""
		Return the likelihood of a note given a state
		
		:param state: a sequence of viewPoints such as len(state) = order
		:param note: integer corresponding to the element

		:type state: np.array(order)
		:type note: int

		:return: float value of the likelihood
		"""

		# in order to work with numpy array and list
		if not isinstance(state, str):
			state = str(list(state))

		if state in self.probabilities:
			pass
		else:
			if self.VERBOSE:
				print("We never saw this state in database.")
			return None

		if str(note) in self.probabilities[state]:
			return self.probabilities[state][str(note)]
		else:
			if self.VERBOSE:
				print("We never saw this transition in database.")
			return 0.0

	def getEntropyMax(self, state):
		"""
		Return the maximum entropy for an alphabet. This is the case where all element is equiprobable.

		:param state: state to compute from
		:type state: list or str(list)

		:return: maxEntropy (float)	
		"""

		alphabetSize = len(self.getPrediction(state).keys())

		maxEntropy = 0

		for i in range(alphabetSize):
			maxEntropy -= (1/alphabetSize) * math.log(1/alphabetSize, 2)

		return maxEntropy

	def getEntropy(self, state):
		"""
		Return shanon entropy of the distribution of the model from a given state

		:param state: state to compute from
		:type state: list or str(list)

		:return: entropy (float)
		"""

		if str(list(state)) in self.entropies:
			return self.entropies[str(list(state))]

		P = self.getPrediction(state).values()

		entropy = 0

		for p in P:
			entropy -= p * math.log(p, 2)

		if not isinstance(state, str):
			state = str(list(state))

		self.entropies[state] = entropy

		return entropy

	def getObservations(self, state):

		if str(list(state)) in self.sum:
			return self.sum[str(list(state))]
		else:
			return None

	def getRelativeEntropy(self, state):
		"""
		Return the relative entropy H(m)/Hmax(m). It is used for weighting the merging of models without bein affected by the alphabet size.

		:param state: state to compute from
		:type state: list or str(list)

		:return: entropy (float)		
		"""

		maxEntropy = self.getEntropyMax(state)

		if maxEntropy > 0:
			return self.getEntropy(state)/maxEntropy

		else:
			return 1


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

	def sample(self, S):
		"""
		Return a element sampled from the model given the sequence S

		:param S: sequence to sample from

		:type S: list of integers

		:return: sampled element (int)
		"""

		state = str(list(S[-self.order:]))

		if DEBUG:
			print("state:", state)
			print("sequence:", S)

		distribution = []
		# We reconstruct the distribution according to the sorting of the alphabet
		for elem in self.alphabet:
			distribution.append(self.getLikelihood(state, elem))

		ret = int(np.random.choice(self.alphabet, p=distribution))

		return ret

	def generate(self, length):
		"""
		Implement a very easy random walk in order to generate a sequence

		:param length: length of the generated sequence (in elements, not beat so it depends on the quantization)
		:type length: int

		:return: sequence (score object) 
		"""

		S = []
		# We uniformly choose the first element
		S.extend(ast.literal_eval(np.random.choice(self.stateAlphabet)))

		while len(S) < length and str(S[-self.order:]) in self.stateAlphabet:

			S.append(self.sample(S))

		return score.score(S)

