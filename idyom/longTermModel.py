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

	def __init__(self, viewPoint, maxOrder=None):

		# ViewPoint to use
		self.viewPoint = viewPoint

		# maximum order if given
		self.maxOrder = maxOrder

	def train(self, data):
		""" 
		Fill the matrix from data
		
		:param data: data to train from

		:type data: list of np.array or list of list of int
		"""
		if isinstance(data, list):
			maxOrder = len(data[0])
			for i in range(1, len(data)):
				maxOrder = max(len(data[i]), maxOrder)
		else:
			maxOrder = len(data)

		if self.maxOrder is None: 
			maxOrder = maxOrder // 2
		else:
			maxOrder = self.maxOrder

		self.maxOrder = maxOrder

		if VERBOSE:
			print("The maximal order is:", self.maxOrder)

		# list contening different order markov chains
		self.models = []
		for order in range(1, self.maxOrder+1):
			self.models.append(markovChain.markovChain(order))


		# training all the models
		for i in tqdm(range(len(self.models))):
			self.models[i].train(data)
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
		P = self.getPrediction(state).values()

		if None in P:
			print(state)
			print(P)
			quit()

		entropy = 0

		for p in P:
			if p != 0:
				entropy -= p * math.log(p, 2)

		return entropy

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
		for model in self.models:
			# we don't want to take in account a model that is not capable of prediction
			if model.order <= len(state) and model.getLikelihood(str(list(state[-model.order:])), note) is not None:
				
				probas.append(model.getLikelihood(state[-model.order:], note))
				weights.append(model.getEntropy(state[-model.order:]))
				observations.append(model.getObservations(state[-model.order:]))

		if probas == [] and False:
			print(state)
			print(len(state))
			print(model.getLikelihood(str(list(state[-model.order:])), note) )
			print(model.order)
			print()

		if probas == []:
			return None
		if False and np.sum(observations) > 0:
			observations = np.array(observations) + 20
			observations = observations / np.sum(observations)
			weights = weights*observations

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
				weights = np.array(weights)
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

