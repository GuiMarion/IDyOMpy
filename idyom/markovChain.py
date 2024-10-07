from idyom import data

import numpy as np
import pickle
from tqdm import tqdm
import ast
import math
import warnings

THRESHOLD = 0
DEBUG = False

class markovChain():
	"""
	Module defining MarkovChain model and usefull functions for the project

	:param alphabetSize: number of elements in the alphabet
	:param VERBOSE: print some strange behoviors, for example if asking for unknwown states

	:type order: int
	:type VERBOSE: bool
	"""
	def __init__(self, order, depth=0, VERBOSE=False, STM=False, evolutive=False):

		# order of the model
		self.order = order

		# depth of the model
		self.depth = depth

		# Wehther it's an evolutive model
		self.evolutive = evolutive

		# store the number of occurences of the transitions
		self.observationsProbas = {}

		# alphabet of state of the data
		self.stateAlphabet = []

		# alphabet of notes of the data
		self.alphabet = []

		self.VERBOSE = VERBOSE

		# In order to store entropy
		self.entropies = {}

		# For tracking
		self.STM = STM

		# We store the number of observation for every state
		self.SUM = {}

		if order < 1:
			raise(ValueError("order should be grater than 0."))

	def __eq__(self, other): 
		if not isinstance(other, markovChain):
			# don't attempt to compare against unrelated types
			return NotImplemented

		if not self.order == other.order :
			print("Different orders")

		elif not self.depth == other.depth:
			print("Different Depth")

		elif not self.SUM == other.SUM:
			print("Different SUM")

		elif not self.alphabet == other.alphabet:
			print("Different alphabet")

		elif not self.stateAlphabet == other.stateAlphabet:
			print("Different stateAlphabet")
			
		elif not self.STM == other.STM:
			print("Different STM")

		elif not self.observationsProbas == other.observationsProbas:
			print("Different Transitions")

		else:
			return True

		return False


	def train(self, dataset, reverse=False, preComputeEntropies=False):
		"""
		Fill the matrix from data, len(data) should be greater than the order.
		
		:param data: pre-processed data to train with
		:type data: data object or list of int
		"""

		if not isinstance(dataset, list) :
			dataset = [dataset]

		self.usedScores = 0

		for data in dataset:
			if len(data) < self.order + self.depth +1:
				#warnings.warn("We cannot train a model with less data than the order of the model, so we skip this data.")
				pass
			else:
				self.usedScores += 1
				# iterating over data
				for i in range(len(data) - self.order - self.depth):
					# case of the reverse prediction, we take a state get the probability to come back to the current state
					if reverse is True:
						state = str(list(data[i+self.order + self.depth : i+self.order*2 + self.depth]))
						target_elem = str(list(data[i:i+self.order])[0])
					else:
						state = str(list(data[i:i+self.order]))
						target_elem = str(list(data[i+self.order + self.depth : i+self.order*2 + self.depth])[0])

					# constructing alphabet
					if state not in self.observationsProbas:
						self.stateAlphabet.append(state)
						self.SUM[state] = 0
						self.observationsProbas[state] = {}

					if target_elem not in self.alphabet:
						self.alphabet.append(target_elem)

					# constructing state to note transitions
					if target_elem not in self.observationsProbas[state]:
						self.observationsProbas[state][target_elem] = 1
					else:
						self.observationsProbas[state][target_elem] += 1

					self.SUM[state] += 1

		# We sort the alphabet
		self.alphabet.sort()
		self.stateAlphabet.sort()

		if preComputeEntropies and self.STM is False:
			for state in self.stateAlphabet:
				self.getEntropy(state)

	def getProbability(self, state, target):
		if state in self.observationsProbas and target in self.observationsProbas[state]:
			return self.observationsProbas[state][target] / self.SUM[state]
		return 0.

	def getProbabilities(self, state):
		probabilities = {}
		for target in self.observationsProbas[state]:
			probabilities[target] = self.observationsProbas[state][target] / self.SUM[state]

		return probabilities


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

		if state in self.observationsProbas:
			return self.getProbabilities(state)
		else:

			if self.VERBOSE:
				print("We never saw this state in database")
			
			dico = {}
			# if we never saw the state, all letter are equilikely
			for z in self.alphabet:
				dico[z] = 1/len(self.alphabet)

			return dico

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

		if state in self.observationsProbas:
			pass
		else:
			if self.VERBOSE:
				print("We never saw this state in database.")
			# as we never saw this state in the database, every note is equiprobable
			if len(self.alphabet) > 0:
				return 1/len(self.alphabet)
			else:
				return None

		if str(note) in self.observationsProbas[state]:
			return self.getProbability(state, str(note))
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


	def getObservationsSum(self):

		ret = 0
		for state in self.SUM:
			ret += self.SUM[state]

		return ret

	def getEntropy(self, state):
		"""
		Return shanon entropy of the distribution of the model from a given state

		:param state: state to compute from
		:type state: list or str(list)

		:return: entropy (float)
		"""
		if not self.evolutive and not self.STM and str(list(state)) in self.entropies:
			return self.entropies[str(list(state))]

		# in order to work with numpy array and list
		if not isinstance(state, str):
			state = str(list(state))

		# if the state was never seen, the entropy is the maximal entropy for |alphabet|
		if state not in self.observationsProbas:
			return -math.log(1/len(self.alphabet))

		P = self.getPrediction(state).values()
		entropy = 0

		for p in P:
			entropy -= p * math.log(p, 2)

		if not self.STM and not self.evolutive:
			self.entropies[state] = entropy

		return entropy

	def getObservations(self, state):

		if str(list(state)) in self.SUM:
			return self.SUM[str(list(state))]
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

	def getMatrix(self):
		"""
		Return the transition matrix between states and notes

		:return: transition matrix (np.array())
		"""

		matrix = np.zeros((len(self.stateAlphabet), len(self.alphabet)))
		k1 = 0
		for state in self.stateAlphabet:
			k2 = 0
			for target in self.alphabet:
				matrix[k1][k2] = self.getProbability(state, target)

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


	# three Methods for mergeProbasPPM	
	def getTotalCount(self, state):
		"""
		Return the total number of observations for a given context, used by mergeProbasPPM in longTermModel

		:param state: a string representation of a list

		:type state: str

		:return: Total count of the context (int)
		"""
		return self.SUM.get(state, 0)

	def getCount(self, state, symbol):
		"""
		Return the count of a specific symbol following a given context, used by mergeProbasPPM in longTermModel

		:param state: string representation of a list
		:param symbol: The target symbol

		:type state: str
		:type symbol: str

		:return: Count of the symbol following the context (int)
		"""
		return self.observationsProbas.get(state, {}).get(symbol, 0)


	def getUniqueSymbolCount(self, state):
		"""
		Return the number of unique symbols observed for a given context, used by mergeProbasPPM in longTermModel

		:param state: string representation of a list
		:type state: str

		:return: Number of unique symbols (int)
		"""
		return len(self.observationsProbas.get(state, {}))
