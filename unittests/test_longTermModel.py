import sys 
sys.path.append('../')

from idyom import longTermModel

import unittest
import numpy as np
import os
import ast

# Maximum order for testing
N = 10
viewPoints = ["pitch"]
"""
We only test for [1, 2, 3, ..., N] % 10 data, so all probabilities equal 1, maybe we can build tests for others sequences.
"""

class markovChain_test(unittest.TestCase):

	def setUp(self):
		"""
		Construct some models for testing
		"""

		self.models = []
		for i in range(1, N):
			for viewPoint in viewPoints:
				self.models.append(longTermModel.longTermModel(viewPoint, i))

	def test_train(self):
		"""
		Fill the matrix from data
		
		:param data: pre-processed data to train with
		:type data: data object
		"""

		X = np.arange(1000) % 10

		for i in range(1, N):
			M = longTermModel.longTermModel("pitch", i)
			M.train(X)

	def test_getPrediction(self):
		"""
		Return the probability distribution of notes from a given state
		
		:param state: a sequence of viewPoints of sier order
		:type state: np.array(order)

		:return: np.array(alphabetSize).astype(float)
		"""

		X = np.arange(1000) % 10

		for i in range(1, N):
			M = longTermModel.longTermModel("pitch", i)
			M.train(X)

		state = [1, 2, 3, 4]

		for model in M.models:
			self.assertEqual(M.getPrediction(state)['5'], 1.0)
			self.assertEqual(M.getPrediction(state)['4'], 0.0)



	def test_getLikelihood(self):
		"""
		Return the likelihood of a note given a state
		
		:param state: a sequence of viewPoints of sier order
		:param note: integer or name of the note

		:type state: np.array(order)
		:type note: int or string

		:return: float value of the likelihood
		"""

		X = np.arange(1000) % 10

		for i in range(1, N):
			M = longTermModel.longTermModel("pitch", i)
			M.train(X)

			alphabet = []
			for model in M.models:
				alphabet.extend(model.alphabet)

			alphabet = list(set(alphabet))
			alphabet.sort()


			for state in alphabet:
				for note in alphabet:

					if (int(state) +1) % 10 == int(note) % 10:
						self.assertEqual(M.getLikelihood([int(state)], note), 1.0)
					else:
						self.assertEqual(M.getLikelihood([int(state)], note), 0.0)


	def test_saveAndLoad(self):
		"""
		Check wether the loaded object is the same as the saved one
		"""

		for i in range(1, N):
			M1 = longTermModel.longTermModel("pitch", i)
			X = np.arange(500) % 10
			M1.train(X)
			M1.save("longterm.s")

			M2 = longTermModel.longTermModel("pitch", 1)
			M2.load("longterm.s")

			os.remove("longterm.s")
			for i in range(len(M1.models)):
				self.assertEqual(M1.models[i].__dict__ , M2.models[i].__dict__)


	def test_sample(self):
		X = np.arange(1000) % 10

		for order in range(2, N):
			M = longTermModel.longTermModel("pitch", order)
			M.train(X)

			for z in M.models[order-2].stateAlphabet:
				state = ast.literal_eval(z)
				s = M.sample(state)
				self.assertEqual(M.getLikelihood(state, s), 1.0)

	def test_generate(self):
		"""
		Implement a very easy random walk in order to generate a sequence

		:param length: length of the generated sequence
		:type length: int

		:return: sequence (np.array()) 
		"""

		X = np.arange(1000) % 10

		for order in range(1, N):
			M = longTermModel.longTermModel("pitch", order)
			M.train(X)

			S = M.generate(10)
			S.sort()
			
			self.assertEqual(S, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
		