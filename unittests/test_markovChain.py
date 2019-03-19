import sys 
sys.path.append('../')

from idyom import markovChain

import unittest
import numpy as np
import os
import ast

# Maximum order for testing
N = 10

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
			self.models.append(markovChain.markovChain(i))

	def test_train(self):
		"""
		Fill the matrix from data
		
		:param data: pre-processed data to train with
		:type data: data object
		"""

		X = np.arange(1000) % 10

		for i in range(1, N):
			M = markovChain.markovChain(i)
			M.train(X)
			T = M.transitions

			for state in T:
				target = str(list((np.arange(ast.literal_eval(state)[-1], ast.literal_eval(state)[-1] + i) + 1) % 10))

				self.assertEqual(T[state][target], 1.0)


		X = []
		for i in range(10):
			X.append(np.arange(1000) % 10 - 1)
			np.random.shuffle(X[i])

		for i in range(1, N):
			M = markovChain.markovChain(i)
			M.train(X)

			for state in M.stateAlphabet:
				self.assertEqual(round(sum(M.getPrediction(state).values()), 2), 1.0)


	def test_getPrediction(self):
		"""
		Return the probability distribution of notes from a given state
		
		:param state: a sequence of viewPoints of sier order
		:type state: np.array(order)

		:return: np.array(alphabetSize).astype(float)
		"""

		X = np.arange(1000) % 10

		for i in range(1, N):
			M = markovChain.markovChain(i)
			M.train(X)
			T = M.probabilities

			for state in T:
				
				target = str((ast.literal_eval(state)[-1] + 1) % 10)

				self.assertEqual(T[state][target], 1.0)

			for state in M.stateAlphabet:
				target = str((ast.literal_eval(state)[-1] + 1) % 10)

				self.assertEqual(M.getPrediction(state)[target], 1.0)



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
			M = markovChain.markovChain(i)
			M.train(X)

			for state in M.stateAlphabet:
				for note in M.alphabet:

					target = str((ast.literal_eval(state)[-1] + 1) % 10)

					if target == note:
						self.assertEqual(M.getLikelihood(state, note), 1.0)
					else:
						self.assertEqual(M.getLikelihood(state, note), 0.0)


	def test_saveAndLoad(self):
		"""
		Check wether the loaded object is the same as the saved one
		"""

		for i in range(1, N):
			M1 = markovChain.markovChain(i)
			X = np.arange(500) % 10
			M1.train(X)
			M1.save("unittest.s")

			M2 = markovChain.markovChain(1)
			M2.load("unittest.s")

			os.remove("unittest.s")

			self.assertEqual(M1.__dict__ , M2.__dict__)


	def test_getStatesMatrix(self):
		"""
		Return the transition matrix between states made from the dictionnary

		:return: transition matrix (np.array())
		"""

		X = np.arange(1000) % 10

		for order in range(1, 2):
			M = markovChain.markovChain(order)
			M.train(X)

			matrix = M.getStatesMatrix()

			for i in range(len(M.stateAlphabet)):
				for j in range(len(M.stateAlphabet)):
					target = str(list((np.arange(ast.literal_eval(M.stateAlphabet[i])[-1], ast.literal_eval(M.stateAlphabet[i])[-1] + order) + 1) % 10))

					if target == M.stateAlphabet[j]:
						self.assertEqual(matrix[i][j], 1.0)

					else:
						self.assertEqual(matrix[i][j], 0.0)


	def test_getMatrix(self):
		"""
		Return the transition matrix between states and notes

		:return: transition matrix (np.array())
		"""

		X = np.arange(1000) % 10

		for order in range(1, 2):
			M = markovChain.markovChain(order)
			M.train(X)

			matrix = M.getStatesMatrix()

			for i in range(len(M.stateAlphabet)):
				for j in range(len(M.alphabet)):
					target = str((ast.literal_eval(M.stateAlphabet[i])[-1] +1) %10)
					if target == M.alphabet[j]:
						self.assertEqual(matrix[i][j], 1.0)

					else:
						self.assertEqual(matrix[i][j], 0.0)


	def test_sample(self):

		X = np.arange(1000) % 10

		for order in range(1, N):
			M = markovChain.markovChain(order)
			M.train(X)

			for z in M.stateAlphabet:
				state = ast.literal_eval(z)
				s = M.sample(state)
				self.assertEqual(M.getLikelihood(z, s), 1.0)

	def test_generate(self):
		"""
		Implement a very easy random walk in order to generate a sequence

		:param length: length of the generated sequence
		:type length: int

		:return: sequence (np.array()) 
		"""

		X = np.arange(1000) % 10

		for order in range(1, N):
			M = markovChain.markovChain(order)
			M.train(X)

			S = list(M.generate(10).getData())
			S.sort()
			
			self.assertEqual(S, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
		