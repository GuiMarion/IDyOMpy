from idyom import data
from idyom import markovChain

import numpy as np
import pickle
from tqdm import tqdm
import math

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

	def __init__(self, viewPoint, maxDepth=None, maxOrder=None):

		# ViewPoint to use
		self.viewPoint = viewPoint

		# maximum order if given
		self.maxOrder = maxOrder

		# maximum depth if given
		self.maxDepth = maxDepth

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
			for i in tqdm(range(len(self.models[depth]))):
				#TEMPORARY
				if depth > 1 and i > 1:
					break
				self.models[depth][i].train(data)
				if self.models[depth][i].usedScores == 0:
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
		for model in self.models[0]:
			alphabet.extend(model.alphabet)

		alphabet = list(set(alphabet))
		alphabet.sort()

		dico = {}

		for z in alphabet:
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
		probas = []
		weights = []
		for model in self.models[0]:
			# we don't want to take in account a model that is not capable of prediction
			if model.order <= len(state) and model.getLikelihood(str(list(state[-model.order:])), note) is not None:
				
				probas.append(model.getLikelihood(state[-model.order:], note))
				weights.append(1 - model.getEntropy(state[-model.order:]))

		# Core of our jump model, computing conditional probabilities
		for depth in range(1, self.maxDepth+1):
			predictions = self.models[depth][0].getPrediction(str(list(state[-1:])))
			if predictions is not None:
				proba = 0
				entropy = 0
				#print(state[-1], note)
				#print(predictions)
				for elem in predictions:
					predictions2 = self.reverse[depth-1].getPrediction(str(list([int(elem)])))
					#print(predictions2)
					#print(elem, note)
					#print("ok",self.reverse[depth-1].getLikelihood([int(elem)], note))
					proba += predictions[elem] * self.reverse[depth-1].getLikelihood([int(elem)], note)
					# We compute the entropy H(X,Y) as sum_{x,y} - log(p(x,y))*p(x,y)
					for elem2 in predictions2:
						entropy -= predictions[elem]*predictions2[elem2] * math.log(predictions[elem] * predictions2[elem2], 2)

				probas.append(proba)
				weights.append(1 - entropy)

				#print(proba, entropy)


		if probas == [] and False:
			print(state)
			print(len(state))
			print(model.getLikelihood(str(list(state[-model.order:])), note) )
			print(model.order)
			print()

		#print(probas)
		#print(weights)
		#print()

		return self.mergeProbas(probas, weights)

	def mergeProbas(self, probas, weights):
		"""
		Merging probabilities from different orders, for now we use arithmetic mean

		:param probas: probabilities to merge
		:param weights: weights for the mean, should be get from normalized entropy

		:type probas: list or numpy array
		:type weights: list or numpy array

		:retur: merged probabilities (float)
		"""

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

