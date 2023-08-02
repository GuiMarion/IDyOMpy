from idyom import data

import numpy as np
import pickle
from tqdm import tqdm
import ast
import math
import warnings

THRESHOLD = 0
DEBUG = False

class markovChainOrder0():
	"""
	Module defining MarkovChain model and usefull functions for the project

	:param alphabetSize: number of elements in the alphabet
	:param VERBOSE: print some strange behoviors, for example if asking for unknwown states

	:type order: int
	:type VERBOSE: bool
	"""
	def __init__(self, depth=0, VERBOSE=False, STM=False, evolutive=False):

		# depth of the model
		self.depth = depth

		# Wehther it's an evolutive model
		self.evolutive = evolutive

		# alphabet of state of the data
		self.stateAlphabet = []

		self.globalCounter = 0

		self.VERBOSE = VERBOSE

		# In order to store entropy
		self.entropies = {}

		# For tracking
		self.STM = STM

		# We store the number of observation for every state
		self.SUM = {}

	def __eq__(self, other): 
		if not isinstance(other, markovChain):
			# don't attempt to compare against unrelated types
			return NotImplemented

		elif not self.depth == other.depth:
			print("Different Depth")

		elif not self.SUM == other.SUM:
			print("Different SUM")

		elif not self.stateAlphabet == other.stateAlphabet:
			print("Different stateAlphabet")
			
		elif not self.STM == other.STM:
			print("Different STM")


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
			self.usedScores += 1
			# iterating over data
			for i in range(len(data) - self.depth):
				state = data[i]

				# constructing alphabet
				if state not in self.SUM:
					self.stateAlphabet.append(state)
					self.SUM[state] = 0

				self.SUM[state] += 1
				self.globalCounter += 1

	def getPrediction(self):
		"""
		Return the probability distribution of notes from a given state
		
		:param state: a sequence of viewPoints such as len(state) = order
		:type state: str(np.array(order))

		:return: dictionary | dico[note] = P(note|state)
	
		"""

		# return a row in the matrix

		dico = {}
		for elem in self.SUM:
			dico[elem] = self.SUM[elem]/self.globalCounter

		return dico

	def getLikelihood(self, note):
		"""
		Return the likelihood of a note given a state
		
		:param state: a sequence of viewPoints such as len(state) = order
		:param note: integer corresponding to the element

		:type state: np.array(order)
		:type note: int

		:return: float value of the likelihood
		"""

		if self.globalCounter == 0:
			return None

		if note not in self.SUM:
			return 0


		return self.SUM[note]/self.globalCounter

	def getEntropyMax(self):
		"""
		Return the maximum entropy for an alphabet. This is the case where all element is equiprobable.

		:param state: state to compute from
		:type state: list or str(list)

		:return: maxEntropy (float)	
		"""

		alphabetSize = len(self.getPrediction().keys())

		maxEntropy = 0

		for i in range(alphabetSize):
			maxEntropy -= (1/alphabetSize) * math.log(1/alphabetSize, 2)

		return maxEntropy


	def getObservationsSum(self):

		return globalCounter

	def getEntropy(self):
		"""
		Return shanon entropy of the distribution of the model from a given state

		:param state: state to compute from
		:type state: list or str(list)

		:return: entropy (float)
		"""


		if self.globalCounter == 0: 
			return None

		entropy = 0
		for elem in self.SUM:
			entropy -= (self.SUM[elem]/self.globalCounter) * math.log(self.SUM[elem]/self.globalCounter, 2)

		return entropy


	def getRelativeEntropy(self):
		"""
		Return the relative entropy H(m)/Hmax(m). It is used for weighting the merging of models without bein affected by the alphabet size.

		:param state: state to compute from
		:type state: list or str(list)

		:return: entropy (float)		
		"""

		maxEntropy = self.getEntropyMax()

		if maxEntropy > 0:
			return self.getEntropy()/maxEntropy

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


