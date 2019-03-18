from idyom import data
from idyom import markovChain

import numpy as np
import pickle

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

	#####	TODO ##### Implement pseudo infinite order
	def __init__(self, viewPoint, maxOrder):

		# Maximal order of the model
		self.maxOrder = maxOrder

		# ViewPoint to use
		self.viewPoint = viewPoint

		# list contening different-order markov chains
		self.models = []
		for order in range(1, maxOrder+1):
			self.models.append(markovChain.markovChain(order))

	def train(self, data):
		""" 
		Fill the matrix from data
		
		:param data: data to train from

		:type data: class data
		"""
		for i in range(len(self.models)):
			self.models[i].train(data)


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
		for model in self.models:
			probas.append(model.getLikelihood(str(state[-model.order:]), note))

		return self.mergeProbas(probas)

	def mergeProbas(self, probas):
		return np.mean(probas)



	def sample(self, state):
		"""
		Return a element sampled from the model given the sequence S

		:param S: sequence to sample from

		:type S: list of integers

		:return: sampled element (int)
		"""

		### IN PROGRESS
		alphabet = []
		for model in self.models:
			alphabet.extend(model.alphabet)

		alphabet = list(set(alphabet))
		alphabet.sort()

		dico = {}

		p = 0
		for z in alphabet:
			p +=  self.getLikelihood(state, z)

		if p != 1:
			print(p)

		### IN PROGRESS


		alphabet = []
		for model in self.models:
			alphabet.extend(model.alphabet)

		alphabet = list(set(alphabet))
		alphabet.sort()

		distribution = []
		# We reconstruct the distribution according to the sorting of the alphabet
		for elem in alphabet:
			distribution.append(self.getLikelihood(state, elem))

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

		while len(S) < length:

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

