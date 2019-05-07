import sys 
sys.path.append('../')

from idyom import jumpModel

import unittest
import numpy as np
import os
import ast
import itertools
from tqdm import tqdm

# Maximum order for testing
N = 10
viewPoints = ["pitch"]
"""
We only test for [1, 2, 3, ..., N] % 10 data, so all probabilities equal 1, maybe 
we can build tests for others sequences.
"""

class jumpModel_test(unittest.TestCase):

	def setUp(self):
		"""
		Construct some models for testing
		"""

		self.models = []
		for depth in range(1, N//2):
			for i in range(1, N):
				for viewPoint in viewPoints:
					self.models.append(jumpModel.jumpModel(viewPoint, maxOrder=i, maxDepth=depth))

	def test_train(self):
		"""
		Fill the matrix from data
		
		:param data: pre-processed data to train with
		:type data: data object
		"""
		X = []
		for i in range(10):
			X.append(np.arange(200) % 10 - 1)
			np.random.shuffle(X[i])

		for depth in range(2, N//2):
			for i in range(N):
				M = jumpModel.jumpModel("pitch", maxOrder=i, maxDepth=depth)
				M.train(X)

				if i == 0:
					M = jumpModel.jumpModel("pitch")
					M.train(X)
				x = X[0]
				for start in tqdm(range(len(x) - 2*N)):
					for end in range(start+i+1, start+i+N):

						alphabet = []
						for model in M.models[0]:
							alphabet.extend(model.alphabet)

						alphabet = list(set(alphabet))
						alphabet.sort()

						p = 0
						for z in alphabet:
							p += M.getLikelihood(x[start:end], z)

						if round(p, 2) != 1:
							print(p, x[start:end])
						self.assertEqual(round(p, 2), 1.0)


	def test_getPrediction(self):
		"""
		Return the probability distribution of notes from a given state
		
		:param state: a sequence of viewPoints of sier order
		:type state: np.array(order)

		:return: np.array(alphabetSize).astype(float)
		"""

		X = np.arange(1000) % 10

		for depth in range(1, 10):
			for i in range(1, 5):
				M = jumpModel.jumpModel("pitch", maxOrder=i, maxDepth=depth)
				M.train(X)

				state = [1, 2, 3, 4, 5]

				for j in range(len(M.models)):
					for model in M.models[j]:
						self.assertEqual(round(model.getPrediction(state[-model.order:])[str((6+j)%10)], 2), 1.0)

				for j in range(len(M.reverse)):
					self.assertEqual(round(M.reverse[j].getPrediction(state[0:M.reverse[j].order])[str((0-j)%10)], 2), 1.0)


				self.assertEqual(round(M.getPrediction(state)['6'], 2), 1.0)



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

		for depth in range(1, N//2):
			for i in range(1, N):
				M = jumpModel.jumpModel("pitch", maxOrder=i, maxDepth=depth)
				M.train(X)

				alphabet = []
				for model in M.models[0]:
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
		for depth in range(1, N//2):
			for i in range(1, N):
				M1 = jumpModel.jumpModel("pitch", maxOrder=i, maxDepth=depth)
				X = np.arange(500) % 10
				M1.train(X)
				M1.save("longterm.s")

				M2 = jumpModel.jumpModel("pitch")
				M2.load("longterm.s")

				os.remove("longterm.s")
				for i in range(len(M1.models)):
					self.assertEqual(M1.models[i][0].__dict__ , M2.models[i][0].__dict__)


	def test_sample(self):
		X = np.arange(1000) % 10

		for depth in range(1, N//2):
			for order in range(2, N):
				M = jumpModel.jumpModel("pitch", maxOrder=order, maxDepth=depth)
				M.train(X)

				for z in M.models[0][0].stateAlphabet:
					state = ast.literal_eval(z)
					s = M.sample(state)
					self.assertEqual(round(M.getLikelihood(state, s), 2), 1.0)

	def test_generate(self):
		"""
		Implement a very easy random walk in order to generate a sequence

		:param length: length of the generated sequence
		:type length: int

		:return: sequence (np.array()) 
		"""

		X = []
		for i in range(10):
			X.append((np.arange(100) + i ) % 10)
		for depth in range(1, N//2):
			for order in range(1, N):
				M = jumpModel.jumpModel("pitch", maxOrder=order, maxDepth=depth)
				M.train(X)

				S = M.generate(400)
				S.sort()
				target = list(np.sort(np.arange(400) % 10))
				self.assertEqual(S, target)
		
#unittest.main()
