from idyom import data
from idyom import markovChain
from idyom import longTermModel
from idyom import score

import numpy as np
from glob import glob
import pickle
import matplotlib.pyplot as plt

class idyom():
	"""
	This module represent the entire model, this is what you want to interact with if you only want to use the model.

	:param maxOrder: maximal order of the model
	:param viewPoints: viewPoint to use, cf. data.getViewPoints()

	:type maxOrder: int
	:type viewPoints: list of strings
	"""
	def __init__(self, maxOrder=None, viewPoints=["pitch", "length"], dataTrain=None, dataTrial=None):

		# viewpoints to use for the model
		self.viewPoints = viewPoints

		# maximal order for the markov chains
		self.maxOrder = maxOrder

		# list of all models for each viewpoints
		self.LTM = []
		for viewPoint in self.viewPoints:
			self.LTM.append(longTermModel.longTermModel(viewPoint, maxOrder))

	def train(self, data):
		"""
		Train the models from data
		
		:param data: data to train from

		:type data: data object
		"""

		k = 0
		for viewPoint in self.viewPoints:
			self.LTM[k].train(data.getData(viewPoint))
			k += 1


	def getLikelihoodfromFile(self, file):
		"""
		Return likelihood over a score
		
		:param folder: file to compute likelihood on 

		:type data: string

		:return: np.array(length)

		"""

		D = data.data()
		D.addFile(file)

		probas = np.ones(D.getSizeofPiece(0))
		probas[0] = 1/len(self.LTM[0].models[0].alphabet)

		for model in self.LTM:
			dat = D.getData(model.viewPoint)[0]
			for i in range(1, len(dat)):
				p = model.getLikelihood(dat[:i], dat[i])
				probas[i] *= p

		return probas

	def getLikelihoodfromData(self, D):

		ret = []

		for d in range(D.getSize()):
			probas = np.ones(D.getSizeofPiece(d))
			probas[0] = 1/len(self.LTM[0].models[0].alphabet)

			for model in self.LTM:
				dat = D.getData(model.viewPoint)[d]
				for i in range(1, len(dat)):
					p = model.getLikelihood(dat[:i], dat[i])
					probas[i] *= p

			ret.append(probas)

		return ret

	def getLikelihoodfromFolder(self, folder):
		"""
		Return likelihood over a all dataset
		
		:param folder: folder to compute likelihood on 

		:type data: string

		:return: a list of np.array(length)
		"""
		ret = []
		for filename in glob(folder + '/**', recursive=True):
			if filename[filename.rfind("."):] in [".mid", ".midi"]:
				ret.append(self.getLikelihoodfromFile(filename))

		return ret

	def sample(self, sequence):
		"""
		Sample the distribution from a given sequence, works only with pitch and length

		:param sequence: sequence of viewpoint data

		:type sequence: list

		:return: sample (int)
		"""

		probas = {}

		sequences = {}

		for model in self.LTM:
			sequences[model.viewPoint] = []

		for elem in sequence:
			for model in self.LTM:
				sequences[model.viewPoint].append(elem[model.viewPoint])

		for model in self.LTM:
			probas[model.viewPoint] = model.getPrediction(sequences[model.viewPoint])

		p = []
		notes = []
		for state1 in probas["pitch"]:
			for state2 in probas["length"]:
				p.append(probas["pitch"][state1]*probas["length"][state2])
				tmp = {}
				tmp["pitch"] = int(state1)
				tmp["length"] = int(state2)
				notes.append(tmp)

		if np.sum(p) == 0:
			return None

		ret = np.random.choice(notes, p=p)

		return ret

	def generate(self, length):
		"""
		Return a piece of music generated using the model; works only with pitch and length.

		:param length: length of the output

		:type length: int

		:return: class piece
		"""

		S = [{"pitch": 74, "length": 24}]

		while len(S) < length and S[-1] is not None:
			S.append(self.sample(S))

		if S[-1] is None:
			S = S[:-1]

		ret = []
		for note in S:
			ret.extend([note["pitch"]]*note["length"])


		return score.score(ret)

	def benchmarkQuantization(self, folder, quantizations=[1,2,4,8,16,24,32,64], train=0.8):

		# We get all the midi files
		files = []
		for filename in glob(folder + '/**', recursive=True):
			if filename[filename.rfind("."):] in [".mid", ".midi"]:
				files.append(filename)

		print("____ PROCESSING THE DATA")

		retMeans = np.zeros(len(quantizations))
		retStd = np.zeros(len(quantizations))
		k = 0
		for quantization in quantizations:

			trainData = data.data(quantization=quantization)
			trainData.addFiles(files[:int(train*len(files))])

			testData = data.data(quantization=quantization)
			testData.addFiles(files[int(train*len(files)):], augmentation=False)

			self.cleanWeights(order=self.maxOrder)
			self.train(trainData)
			
			tmp = self.getLikelihoodfromData(testData)
			means = np.zeros(testData.getSize())

			for i in range(len(tmp)):
				means[i] = np.mean(tmp[i])

			retMeans[k] = np.mean(means)
			retStd[k] = np.std(means)
			k += 1
		
		plt.plot(retMeans)
		plt.fill_between(range(len(retMeans)), retMeans + retStd, retMeans - retStd, alpha=.5)
		plt.show()

		return (retMeans, retStd)

	def benchmarkOrder(self, folder, maxOrder, train=0.8):

		# We get all the midi files
		files = []
		for filename in glob(folder + '/**', recursive=True):
			if filename[filename.rfind("."):] in [".mid", ".midi"]:
				files.append(filename)

		print("____ PROCESSING THE DATA")

		trainData = data.data()
		trainData.addFiles(files[:int(train*len(files))])

		testData = data.data()
		testData.addFiles(files[int(train*len(files)):], augmentation=False)

		retMeans = np.zeros(maxOrder)
		retStd = np.zeros(maxOrder)

		for order in range(1, maxOrder):
			self.cleanWeights(order=order)
			self.train(trainData)
			
			tmp = self.getLikelihoodfromData(testData)
			means = np.zeros(testData.getSize())

			for i in range(len(tmp)):
				means[i] = np.mean(tmp[i])

			retMeans[order] = np.mean(means)
			retStd[order] = np.std(means)
		
		plt.plot(retMeans)
		plt.fill_between(range(len(retMeans)), retMeans + retStd, retMeans - retStd, alpha=.5)
		plt.show()

		return (retMeans, retStd)

	def cleanWeights(self, order=None):
		"""
		Delete all trained models and fix an order if given
		"""

		if order is None:
			order = self.maxOrder

		self.LTM = []
		for viewPoint in self.viewPoints:
			self.LTM.append(longTermModel.longTermModel(viewPoint, order))

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
