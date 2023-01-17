from idyom import data
from idyom import markovChain

import numpy as np
import pickle
from tqdm import tqdm
import math

VERBOSE = False	

class longTermModel():
	"""
	Module implementing the Long Term Model from IDyOM, this model contains several Markov Chains of different orders weighted by their respective shanon entropy.

	:param viewPoint: viewPoint to use, cf. data.getViewPoints()
	:param maxOrder: maximal order of the models
	:param alphabetSize(optional): size of the alphabet, number of viewPoints value to take account in

	:type viewPoint: string
	:type maxOrder: int
	:type alphabetSize(optional): int
	"""

	def __init__(self, viewPoint, maxOrder=None, STM=False, init=None, evolutive=False):

		# ViewPoint to use
		self.viewPoint = viewPoint

		# maximum order if given
		self.maxOrder = maxOrder

		# to track if is LTM or STM
		self.STM = STM

		#Wether it's an evolutive model
		self.evolutive = evolutive

		# in order to compute model entropy directly from MC entropies
		self.entropies = {}

		if init is not None:

			maxOrder = len(init)

			if self.maxOrder is None: 
				maxOrder = maxOrder // 2 # CHANGE IT TO maxOrder - 1, maybe
			else:
				maxOrder = self.maxOrder

			self.maxOrder = maxOrder

			if VERBOSE:
				print("The maximal order is:", self.maxOrder)

		# list contening different order markov chains
		self.models = []
		for order in range(1, self.maxOrder+1):
			self.models.append(markovChain.markovChain(order, STM=self.STM, evolutive=evolutive))

		self.benchmark = [0, 0, 0]

	def getObservations(self):
		ret = 0
		for model in self.models:
			ret += model.getObservationsSum()
		return ret

	def train(self, data, shortTerm=False, preComputeEntropies=False):
		""" 
		Fill the matrix from data
		
		:param data: data to train from

		:type data: list of np.array or list of list of int
		"""

		if shortTerm is True:
			# training all the models
			for i in range(len(self.models)):
				self.models[i].train([data[0][-self.models[i].order-1:]])
				if self.models[i].usedScores == 0:
					if VERBOSE:
						print("The order is too high for these data, we stop the training here.")
					break
			return

		# if isinstance(data, list):
		# 	maxOrder = len(data[0])
		# 	for i in range(1, len(data)):
		# 		maxOrder = max(len(data[i]), maxOrder)
		# else:
		# 	maxOrder = len(data)

		# if self.maxOrder is None: 
		# 	maxOrder = maxOrder // 2
		# else:
		# 	maxOrder = self.maxOrder

		# self.maxOrder = maxOrder

		if VERBOSE:
			print("The maximal order is:", self.maxOrder)
		import time
		# training all the models
		for i in range(len(self.models)):
			self.models[i].train(data, preComputeEntropies=preComputeEntropies)
			if self.models[i].usedScores == 0:
				if VERBOSE:
					print("The order is too high for these data, we stop the training here.")
				break

	def getPrediction(self, sequence):
		"""
		Returns the probability distribution from a given state
		
		:param sequence: a sequence of viewPoint data, cf. data.getData(viewPoint)

		:type sequence: np.array(length)

		:return: dictionary | dico[z] = P(z|sequence) (float)
		"""

		alphabet = []
		for model in self.models:
			alphabet.extend(model.alphabet)

		alphabet = list(set(alphabet))
		alphabet.sort()

		dico = {}

		for z in alphabet:
			dico[str(z)] = self.getLikelihood(sequence, z)

		return dico

	def getEntropyMax(self, state):
		"""
		Return the maximum entropy for an alphabet. This is the case where all element is equiprobable.

		:param state: state to compute from
		:type state: list or str(list)

		:return: maxEntropy (float)	
		"""

		alphabetSize = np.count_nonzero(list(self.getPrediction(state).values()))

		maxEntropy = 0

		for i in range(alphabetSize):
			maxEntropy -= (1/alphabetSize) * math.log(1/alphabetSize, 2)

		return maxEntropy

	def getAlphabet(self):
		alphabet = []
		for model in self.models:
			alphabet.extend(model.alphabet)

		return list(set(alphabet))

	def getEntropy(self, state):
		"""
		Return shanon entropy of the distribution of the model from a given state

		:param state: state to compute from
		:type state: list or str(list)

		:return: entropy (float)
		"""
		return self.mergeProbas(self.entropies[str(state)], self.entropies[str(state)]) 
		# P = self.getPrediction(state).values()

		# if None in P:
		# 	print("It's not possible to compute this entropy, we kill the execution.")
		# 	print("state:",state)
		# 	print("probabilities:", P)
		# 	quit()

		# entropy = 0

		# for p in P:
		# 	if p != 0:
		# 		entropy -= p * math.log(p, 2)

		# state = str(state)
		# weights = self.entropies[state]
		# # we inverse the entropies
		# weights = (weights.astype(float)+np.finfo(float).eps)**(-1)
		
		# # Doomy normalization
		# for w in weights:
		# 	if w < 0:
		# 		weights += abs(min(weights))
		# 		break
		# if np.sum(weights) == 0:
		# 	weights = np.ones(len(weights))

		# weights = weights/np.sum(weights)


		# disjoint =self.mergeProbas(self.entropies[state], self.entropies[state]) - np.sum(weights*np.log2(weights))

		# newEntropy = self.mergeProbas(self.entropies[state], self.entropies[state]) + disjoint*0.0

		# self.benchmark[0] += (entropy - newEntropy)**2
		# self.benchmark[1] += 1
		# self.benchmark[2] += entropy

		# return newEntropy

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



	def getLikelihood(self, state, note):
		"""
		Returns the likelihood of a note given a state
		
		:param state: a sequence of viewPoint data, cf. data.getData(viewPoint)
		:param note: the interger or name of the note

		:type state: np.array(length)
		:type note:	int or string

		:return: float value of the likelihood
		"""
		probas = []
		weights = []
		observations = []

		k = -1
		for model in self.models:
			k += 1
			# we don't want to take in account a model that is not capable of prediction
			if model.order <= len(state) and model.getLikelihood(str(list(state[-model.order:])), note) is not None:
				if model.getObservations(state[-model.order:]) is not None: 		
					probas.append(model.getLikelihood(state[-model.order:], note))
					weights.append(model.getRelativeEntropy(state[-model.order:]))
					observations.append(model.getObservations(state[-model.order:]))

		if probas == []:
			return None

		if False and self.mergeProbas(probas, np.array(weights)) < 0.00001:
			print("probas:", probas)
			print("weights:", weights)
			print("note:", note)
			print()

		self.entropies[str(state)] = np.array(weights)

		return self.mergeProbas(probas, np.array(weights))

	def mergeProbas(self, probas, weights, b=1):
		"""
		Merging probabilities from different orders, for now we use arithmetic mean

		:param probas: probabilities to merge
		:param weights: weights for the mean, should be get from normalized entropy

		:type probas: list or numpy array
		:type weights: list or numpy array

		:return: merged probabilities (float)
		"""

		# we inverse the entropies
		weights = (weights.astype(float)+np.finfo(float).eps)**(-b)
		
		# Doomy normalization
		for w in weights:
			if w < 0:
				weights += abs(min(weights))
				break
		if np.sum(weights) == 0:
			weights = np.ones(len(weights))

		weights = weights/np.sum(weights)

		ret = 0
		for i in range(len(probas)):
			ret += probas[i]*weights[i]

		return ret



	def sample(self, state):
		"""
		Return a element sampled from the model given the sequence S

		:param S: sequence to sample from

		:type S: list of integers

		:return: sampled element (int)
		"""


		alphabet = []
		for model in self.models:
			alphabet.extend(model.alphabet)

		alphabet = list(set(alphabet))
		alphabet.sort()

		distribution = []
		# We reconstruct the distribution according to the sorting of the alphabet
		for elem in alphabet:
			distribution.append(self.getLikelihood(state, elem))

		#print(state)
		#print(np.sum(distribution))

		ret = int(np.random.choice(alphabet, p=distribution))

		return ret

	def generate(self, length):
		"""
		Implement a very easy random walk in order to generate a sequence

		:param length: length of the generated sequence (in elements, not beat so it depends on the quantization)
		:type length: int

		:return: sequence (np.array()) 
		"""

		S = []
		# We uniformly choose the first element
		S.append(int(np.random.choice(self.models[0].alphabet)))

		while len(S) < length and str([S[-1]]) in self.models[0].stateAlphabet :

			S.append(self.sample(S))

		return S

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

