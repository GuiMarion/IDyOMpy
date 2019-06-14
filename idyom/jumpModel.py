from idyom import data
from idyom import markovChain

import numpy as np
import pickle
from tqdm import tqdm
import math
import time

VERBOSE = False

class jumpModel():
	"""
	Module implementing the Jump Model by Guilhem Marion, this model contains several Markov Chains of different orders and depth allowing to gather information by expecting notes further in the future.
	The models are averaged and weighted by their respective shanon entropy.

	:param viewPoint: viewPoint to use, cf. data.getViewPoints()
	:param maxOrder: maximal order of the models
	:param maxDepth: maximal depth to use
	:param alphabetSize(optional): size of the alphabet, number of viewPoints value to take account in

	:type viewPoint: string
	:type maxOrder: int
	:type maxDepth: int
	:type alphabetSize(optional): int
	"""

	def __init__(self, viewPoint, maxDepth=10, maxOrder=None):

		# ViewPoint to use
		self.viewPoint = viewPoint

		# maximum order if given
		self.maxOrder = maxOrder

		# maximum depth if given
		self.maxDepth = maxDepth

		# In order to store the entropy 
		self.entropies = {}

		# In order to store likelihoods
		self.likelihoods = {}

	def train(self, data):
		""" 
		Fill the matrix from data
		
		:param data: data to train from

		:type data: list of np.array or list of list of int
		"""
		if isinstance(data, list):
			maxOrder = len(data[0])
			for i in range(1, len(data)):
				maxOrder = min(len(data[i]), maxOrder)
		else:
			maxOrder = len(data)

		if self.maxOrder is None: 
			maxOrder = maxOrder//2
		else:
			maxOrder = self.maxOrder

		self.maxOrder = maxOrder
		if VERBOSE:
			print("The maximal order is:", self.maxOrder)

		# list contening different order markov chains
		self.models = []
		for depth in range(self.maxDepth+1):
			self.models.append([])
			for order in range(1, self.maxOrder+1):
				self.models[depth].append(markovChain.markovChain(order, depth=depth))


		self.reverse =[]
		for depth in range(self.maxDepth):
			self.reverse.append(markovChain.markovChain(1, depth=depth))
			self.reverse[depth].train(data, reverse=True)

		# training all the models
		for depth in range(self.maxDepth+1):
			for order in range(maxOrder):
				self.models[depth][order].train(data)
				if self.models[depth][order].usedScores == 0:
					if VERBOSE:
						print("The order is too high for these data, we stop the training here.")
					break

		self.alphabet = []
		for model in self.models[0]:
			self.alphabet.extend(model.alphabet)

		self.alphabet = list(set(self.alphabet))
		self.alphabet.sort()


	def getPrediction(self, sequence):
		"""
		Returns the probability distribution from a given state
		
		:param sequence: a sequence of viewPoint data, cf. data.getData(viewPoint)

		:type sequence: np.array(length)

		:return: dictionary | dico[z] = P(z|sequence) (float)
		"""
		
		dico = {}

		for z in self.alphabet:
			dico[str(z)] = self.getLikelihood(sequence, z)

		return dico


	def getLikelihood(self, state, note):
		"""
		Returns the likelihood of a note given a state
		
		:param state: a sequence of viewPoint data, cf. data.getData(viewPoint)
		:param note: the interger or name of the note

		:type state: np.array(length)
		:type note:	int or string

		:return: float value of the likelihood
		"""

		identi = str(list(state)) + str(note)
		if identi in self.likelihoods:
			return self.likelihoods[identi]

		# Model with no depth
		probas = []
		weights = []
		for model in self.models[0]:
			# we don't want to take in account a model that is not capable of prediction
			lkh = model.getLikelihood(str(list(state[-model.order:])), note)
			if model.order <= len(state) and lkh is not None:
				
				probas.append(lkh)
				weights.append(model.getEntropy(state[-model.order:]))


		# Core of our jump model, computing conditional probabilities
		for depth in range(1, self.maxDepth):

			# we compute the distribution from the current states for jumps of size depth
			predictions_all = []
			entropy_all = []
			for order in range(self.maxOrder):
				if order+1 <= len(state) and self.models[depth][order].getLikelihood(str(list(state[-order-1:])), note) is not None:
					predictions_all.append(self.models[depth][order].getPrediction(str(list(state[-order-1:]))))
					entropy_all.append(self.models[depth][order].getRelativeEntropy(state[-order-1:]))

			# We merge these distributions over all orders
			predictions = self.mergeProbas(predictions_all, entropy_all, mergeOrders=True, b=10)

			if predictions is not None:
				proba = 0
				entropy = 0

				for elem in predictions:
					predictions2 = self.reverse[depth-1].getPrediction(str(list([int(elem)])))
					if predictions2 is not None:
						proba += predictions[elem] * self.reverse[depth-1].getLikelihood([int(elem)], note)
						# We compute the entropy H(X,Y) as sum_{x,y} - log(p(x,y))*p(x,y)
						for elem2 in predictions2:
							if predictions[elem] != 0:
								entropy -= predictions[elem]*predictions2[elem2] * math.log(predictions[elem] * predictions2[elem2], 2)

				probas.append(proba)
				weights.append(entropy)

		if probas == [] and False:
			print(state)
			print(len(state))
			print(model.getLikelihood(str(list(state[-model.order:])), note) )
			print(model.order)
			print()

		ret = self.mergeProbas(probas, weights, b=0.5)

		self.likelihoods[identi] = ret

		return ret 

	def getReverse(self, note, target, depth):
		"""
		Returns the likelihood of the reverse by merging different techniques

		:param note: starting note
		:param target: note we want to go to

		:type probas: string
		:type weights: string

		:return: merged probabilities (float)		
		"""

		probas = []
		weights = []

		# Direct path
		probas.append(self.reverse[depth-1].getLikelihood([int(note)], target))
		weights.append(self.reverse[depth-1].getEntropy([int(note)]))

		return self.mergeProbas(probas, weights, b=1)

	def mergeProbas(self, probas, weights, mergeOrders=False, b=1, thr=5):
		"""
		Merging probabilities from different orders, for now we use arithmetic mean

		:param probas: probabilities to merge
		:param weights: weights for the mean, should be get from normalized entropy

		:type probas: list or numpy array
		:type weights: list or numpy array

		:return: merged probabilities (float)
		"""
		
		weights = np.array(weights)
		probas = np.array(probas)

		# we inverse the entropies
		weights = (np.array(weights).astype(float)+np.finfo(float).eps)**(-b)
		
		# Doomy normalization
		for w in weights:
			if w < 0:
				weights = np.array(weights)
				weights += abs(min(weights))
				break
		if np.sum(weights) == 0:
			weights = np.ones(len(weights))

		weights = weights/np.sum(weights)

		# if we want to merge predictions(distribution for all z in sigma) and not probabilities
		if mergeOrders:
			alphabet = []
			for d in probas:
				for key in d:
					if key not in alphabet:
						alphabet.append(key)

			ret = {}
			for elem in alphabet:
				ret[elem] = 0
				for i in range(len(probas)):
					if elem in probas[i]:
						ret[elem] += probas[i][elem] * weights[i]
			return ret


		# if we want te merge probabilities
		ret = 0
		for i in range(len(probas)):
			ret += probas[i]*weights[i]

		return ret

	def getEntropyMax(self, state):
		"""
		Return the maximum entropy for an alphabet. This is the case where all element is equiprobable.

		:param state: state to compute from
		:type state: list or str(list)

		:return: maxEntropy (float)	
		"""

		alphabetSize = np.count_nonzero(self.getPrediction(state).values())

		maxEntropy = 0

		for i in range(alphabetSize):
			maxEntropy -= (1/alphabetSize) * math.log(1/alphabetSize, 2)

		return maxEntropy


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
			if p != 0:
				entropy -= p * math.log(p, 2)

		self.entropies[str(list(state))] = entropy

		return entropy

	def sample(self, state):
		"""
		Return a element sampled from the model given the sequence S

		:param S: sequence to sample from

		:type S: list of integers

		:return: sampled element (int)
		"""


		alphabet = []
		for model in self.models[0]:
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
		S.append(int(np.random.choice(self.models[0][0].alphabet)))

		while len(S) < length and str([S[-1]]) in self.models[0][0].stateAlphabet :

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

