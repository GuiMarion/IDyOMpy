from idyom import data
from idyom import markovChain
from idyom import longTermModel

import numpy as np
from glob import glob
import pickle
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import math

class idyom():
	"""
	This module represent the entire model, this is what you want to interact with if you only want to use the model.

	:param maxOrder: maximal order of the model
	:param viewPoints: viewPoint to use, cf. data.getViewPoints()

	:type maxOrder: int
	:type viewPoints: list of strings
	"""
	def __init__(self, maxOrder=24, viewPoints=["pitch", "length"], dataTrain=None, dataTrial=None, stm=True, evolutive=False, use_original_PPM=False):

		# viewpoints to use for the model
		self.viewPoints = viewPoints

		# maximal order for the markov chains
		self.maxOrder = maxOrder

		# whether we also use short term model or not
		self.stm = stm

		self.evolutive = evolutive

		self.use_original_PPM = use_original_PPM

		# list of all models for each viewpoints
		self.LTM = []
		for viewPoint in self.viewPoints:
			self.LTM.append(longTermModel.longTermModel(viewPoint, maxOrder=self.maxOrder, evolutive=evolutive, use_original_PPM=self.use_original_PPM))

	def train(self, data, preComputeEntropies=False):
		"""
		Train the models from data
		
		:param data: data to train from

		:type data: data object
		"""

		k = 0
		for viewPoint in self.viewPoints:
			self.LTM[k].train(data.getData(viewPoint), preComputeEntropies=preComputeEntropies)
			k += 1

	def mergeProbas(self, probas, weights, b=1):
		"""
		Merging probabilities from different models, for now we use arithmetic mean

		:param probas: probabilities to merge
		:param weights: weights for the mean, should be get from normalized entropy

		:type probas: list or numpy array
		:type weights: list or numpy array

		:return: merged probabilities (float)
		"""
		# print(probas)
		weights = np.array(weights) + 0.01
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
		probas = np.array(probas)

		# arithmetic mean for PPM
		if self.use_original_PPM:
			ret = np.sum(probas * weights)
		# weighted geometric mean for py, with small delta values kicked out
		else:
			if len(probas) > 1 and probas[1] < 1e-15:
				probas = [probas[0]]
				weights = [weights[0]]
			ret = np.prod([p ** w for p, w in zip(probas, weights)])
		return ret

	def getLikelihoodfromFile(self, file, short_term_only=False, long_term_only=False):
		"""
		Return likelihood over a score with the genuine entropy computation (not the approx)
		
		:param folder: file to compute likelihood on 

		:type data: string

		:return: np.array(length)

		"""
		D = data.data()
		D.addFile(file)

		probas = np.ones((D.getSizeofPiece(0), len(self.viewPoints)))
		entropies = np.zeros((D.getSizeofPiece(0), len(self.viewPoints)))

		for model in self.LTM:
			if model.viewPoint == "pitch":
				probas[0,0] = model.modelOrder0.getLikelihood(D.getData(model.viewPoint)[0][0])
				entropies[0,0] = model.modelOrder0.getEntropy()

		v = 0
		for model in self.LTM:
			dat = D.getData(model.viewPoint)[0]
			for i in range(1, len(dat)):
				# Depending if the viewpoint is direct effect (pitch) 
				# or derived (length, interval) we need to delay the timing
				if model.viewPoint in ["length", "interval"]:
					add = 1
				else:
					add = 0
				probas[i, v] = self.getLikelihood(model, dat[:i-add], dat[i-add], short_term_only=short_term_only, long_term_only=long_term_only)
				entropies[i, v] = self.getEntropy(model, dat[:i-add], short_term_only=short_term_only, long_term_only=long_term_only)
			v += 1

		probas = np.prod(probas, axis=1)
		entropies = np.sum(entropies, axis=1)

		return probas, entropies

	def getLikelihoodfromFileSimplifiedEntropies(self, file, short_term_only=False, long_term_only=False):
		"""
		Return likelihood over a score (using entropy approximation)
		
		:param folder: file to compute likelihood on 

		:type data: string

		:return: np.array(length)

		"""

		D = data.data()
		D.addFile(file)

		probas = np.ones(D.getSizeofPiece(0))
		entropies = np.zeros(D.getSizeofPiece(0))

		for model in self.LTM:
			if model.viewPoint == "pitch":
				probas[0] = model.modelOrder0.getLikelihood(D.getData(model.viewPoint)[0][0])
				entropies[0] = model.modelOrder0.getEntropy()

		for model in self.LTM:
			dat = D.getData(model.viewPoint)[0]
			if long_term_only is False:
				# we instanciate a Short Term Model for the current viewpoint
				STM = longTermModel.longTermModel(model.viewPoint, maxOrder=self.maxOrder, STM=True, init=dat, use_original_PPM=self.use_original_PPM)

			for i in range(1, len(dat)):
				# If the viewpoint is relative we delay the effect of 1 note
				if model.viewPoint in ["length", "interval"]:
					i = i - 1

				if long_term_only is False:
					STM.train([dat[:i]], shortTerm=True)

				p1 = model.getLikelihood(dat[:i], dat[i])
				if p1 is None:
					e1 = 4.9
				else:
					e1 = model.getEntropy(dat[:i])

				flag = True

				# This happens when the state never happened in the training data
				if p1 is None:
					p1 = 1/30
					e1 = 4.9
					flag = None
				if long_term_only is False and dat[:i] != []:
					p2 = STM.getLikelihood(dat[:i], dat[i])
					if p2 is None:
						e2 = 4.9
					else:
						e2 = STM.getEntropy(dat[:i])
				if long_term_only:
					p = p1
					e = e1
				elif short_term_only:
					p = p2
					e = e2
					if p is None:
						p = 1/30
						e = 4.9
				elif self.stm and p2 is not None:
					if flag is not None:
						p = self.mergeProbas([p1, p2], [model.getRelativeEntropy(dat[:i]), STM.getRelativeEntropy(dat[:i])])
						e = self.mergeProbas([e1, e2], [model.getRelativeEntropy(dat[:i]), STM.getRelativeEntropy(dat[:i])])
					else:
						p = p2
						e = e2
				else:
					p = p1
					e = e1

				# If the viewpoint is relative we delay the effect of 1 note
				if model.viewPoint in ["length", "interval"]:
					i = i + 1

				# We do not want to use the last element of a relative viewpoint (it is never 'heard')
				if i < len(probas):
					probas[i] *= p
					entropies[i] += e


		return probas, entropies


	def getLikelihood(self, model, context, nextNote, short_term_only=False, long_term_only=False):
		"""
		Return likelihood of a not given a context
		
		:param model: the idyom model to use

		:param context: the sequence of previous notes

		:param nextNote: the note to compute the likelihood of

		:return: likelihood as a float

		"""
		if long_term_only is True or context == []: 
			return model.getLikelihood(context, nextNote)

		STM = longTermModel.longTermModel(model.viewPoint, maxOrder=self.maxOrder, STM=True, init=context, use_original_PPM=self.use_original_PPM)
		STM.train([context], shortTerm=True) # True??
		p2 = STM.getLikelihood(context, nextNote)
		if p2 is None: 
			p2 = 1/30

		if short_term_only is True:
			return p2

		p1 = model.getLikelihood(context, nextNote)
		flag = True

		# This happens when the state never happened in the training data it is very very unlikely (even not possible in this implementation)
		if p1 is None:
			p1 = 1/30
			flag = None

		if p2 is not None:
			if flag is not None:
				p = self.mergeProbas([p1, p2], [model.getRelativeEntropy(context, genuine_entropies=True), STM.getRelativeEntropy(context, genuine_entropies=True)])
			else:
				p = p2
		else:
			p = p1

		return p

	def getEntropy(self, model, context, short_term_only=False, long_term_only=False):
		"""
		Return the entropy (genuine, not approximated) given a context
		
		:param model: the idyom model to use

		:param context: the sequence of previous notes

		:return: entropy as a float

		"""
		entropy = 0
		for note in model.modelOrder0.stateAlphabet:
			p = self.getLikelihood(model, context, note, short_term_only=short_term_only, long_term_only=long_term_only)
			entropy += p*math.log(p, 2)
		return -entropy

	def getDistributionsfromFile(self, file, threshold, short_term_only=False, long_term_only=False, normalization=True, genuine_entropies=False):
		"""
		Return likelihood over a score
		
		:param folder: file to compute likelihood on 

		:type data: string

		:return: np.array(length)

		"""

		D = data.data()
		D.addFile(file)

		distribution = []

		for model in self.LTM:
			if model.viewPoint == "length":
				dat = D.getData(model.viewPoint)[0]
				STM = longTermModel.longTermModel(model.viewPoint, maxOrder=20, STM=True, init=dat, use_original_PPM=self.use_original_PPM)

				for i in tqdm(range(1, len(dat))):
					# we instanciate a Short Term Model for the current viewpoint

					STM.train([dat[:i]], shortTerm=True)
					predictions_LTM = model.getPrediction(dat[:i])
					predictions_STM = STM.getPrediction(dat[:i])

					durations = []
					for duration in predictions_LTM:
						if duration not in durations and predictions_LTM[duration] != 0:
							durations.append(duration)

					for duration in predictions_STM:
						if duration not in durations and predictions_STM[duration] != 0:
							durations.append(duration)

					distribution_note = {}
					for duration in durations:
						if duration in predictions_LTM:
							p1 = predictions_LTM[duration]
							flag = True
						else:
							p1 = 1/30
							flag = None
						if duration in predictions_STM:
							p2 = predictions_STM[duration]
						else:
							p2 = None

						if self.stm and p2 is not None:
							if flag is not None:
								p = self.mergeProbas([p1, p2], [model.getRelativeEntropy(dat[:i]), STM.getRelativeEntropy(dat[:i])])
							else:
								p = p2
						else:
							p = p1

						if long_term_only:
							p = p1
						if short_term_only:
							p = p2
							if p is None:
								p = 1/30
						distribution_note[duration] = p

					distribution.append(distribution_note)


		### Time Representation

		D = data.data()
		D.addFile(file)

		probas, entropies = self.getLikelihoodfromFile(file, short_term_only=short_term_only, long_term_only=long_term_only)

		# We compute the surprise by using -log2(probas)
		# probas = -np.log(probas+sys.float_info.epsilon)/np.log(2)

		# We get the length of the notes
		lengths = D.getData("length")[0]

		ret = []
		for i in range(len(probas)):
			ret.append(probas[i])
			for j in range(int(lengths[i])):
				ret.append(0)

		notes_surprise = ret


		indexes = []
		probas = []
		current_index = 1
		for i in range(len(distribution)):
			sum_distribution = sum(distribution[i].values())
			keys = np.array(list(distribution[i])).astype(int)
			keys.sort()
			for duration in keys:
				duration = str(duration)
				if int(duration) < int(lengths[i]) and distribution[i][duration]/sum_distribution > threshold:
					indexes.append(current_index+int(duration))
					probas.append(distribution[i][duration]/sum_distribution)

				if normalization:
					sum_distribution -= distribution[i][duration]
			current_index += int(lengths[i]) +1


		missing_notes = np.zeros(len(notes_surprise))
		missing_notes[indexes] = probas

		indexes = []
		probas = []
		current_index = 1
		for i in range(len(distribution)):
			sum_distribution = sum(distribution[i].values())
			keys = np.array(list(distribution[i])).astype(int)
			keys.sort()
			for duration in keys:
				duration = str(duration)
				if int(duration) == int(lengths[i]) and distribution[i][duration]/sum_distribution > threshold:
					indexes.append(current_index+int(duration))
					probas.append(distribution[i][duration]/sum_distribution)

				if normalization:
					sum_distribution -= distribution[i][duration]
			current_index += int(lengths[i]) +1


		actual_notes = np.zeros(len(notes_surprise))
		actual_notes[indexes] = probas

		return actual_notes, missing_notes

	def getDistributionsfromFolder(self, folder, threshold, zero_padding=True, time_representation=False, short_term_only=False, long_term_only=False):
		"""
		Return likelihood over a all dataset
		
		:param folder: folder to compute likelihood on 
		:param zero_padding: return surprise as spikes if True

		:type data: string
		:type zero_padding: bool

		:return: a list of np.array(length)
		"""

		ret = []
		files = []
		for filename in tqdm(glob(folder + '/**', recursive=True)):
			if filename[filename.rfind("."):] in [".mid", ".midi"]:
				ret.append(self.getDistributionsfromFile(filename, threshold))
				files.append(filename)

		return ret, files

	def getSurprisefromFile(self, file, zero_padding=False, time_representation=False, short_term_only=False, long_term_only=False, genuine_entropies=False):
		"""
		Return surprise(-log2(p)) over a score
		
		:param folder: file to compute surprise on 
		:param zero_padding: return surprise as spikes if True

		:type data: string
		:type zero_padding: bool

		:return: list of float

		"""

		if genuine_entropies is False:
			# We use the approx
			probas, entropies = self.getLikelihoodfromFileSimplifiedEntropies(file, short_term_only=short_term_only, long_term_only=long_term_only)
		else:
			# We compute the genuine entropies (takes 5 times longer)
			probas, entropies = self.getLikelihoodfromFile(file, short_term_only=short_term_only, long_term_only=long_term_only)

		# We compute the surprise by using -log2(probas)
		probas = -np.log(probas+sys.float_info.epsilon)/np.log(2)

		if time_representation is False:
			return probas, entropies

		D = data.data()
		D.addFile(file)
		# We get the length of the notes
		lengths = D.getData("length")[0]

		surprise = []
		entropy = []
		for i in range(len(probas)):
			surprise.append(probas[i])
			entropy.append(entropies[i])
			for j in range(int(lengths[i])):
				if zero_padding:
					surprise.append(0)
					entropy.append(0)
				else:
					surprise.append(probas[i])
					entropy.append(entropies[i])

		return surprise, entropy

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

	def getSurprisefromFolder(self, folder, zero_padding=True, time_representation=False, short_term_only=False, long_term_only=False, genuine_entropies=False):
		"""
		Return likelihood over a all dataset
		
		:param folder: folder to compute likelihood on 
		:param zero_padding: return surprise as spikes if True

		:type data: string
		:type zero_padding: bool

		:return: a list of np.array(length)
		"""
		likelihoods = []
		entropies = []
		files = []
		for filename in glob(folder + '/**', recursive=True):
			if filename[filename.rfind("."):] in [".mid", ".midi"]:
				L, E = self.getSurprisefromFile(filename, time_representation=time_representation, \
					zero_padding=zero_padding, short_term_only=short_term_only, long_term_only=long_term_only, genuine_entropies=genuine_entropies)
				likelihoods.append(L)
				entropies.append(E)
				files.append(filename)

		return likelihoods, entropies, files


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
				if probas["pitch"][state1] is not None and probas["length"][state2] is not None:
					p.append(probas["pitch"][state1]*probas["length"][state2])
					tmp = {}
					tmp["pitch"] = int(state1)
					tmp["length"] = int(state2)
					notes.append(tmp)

		if np.sum(p) == 0:
			return None

		if np.sum(p) != 1:
			print(np.sum(p))
			p = p/np.sum(p)

		ret = np.random.choice(notes, p=p)

		return ret


	def benchmarkQuantization(self, folder, quantizations=[1,2,3,4,5,6,7,8,10,12,16,24,32,64], train=0.8):

		# We get all the midi files
		files = []
		for filename in glob(folder + '/**', recursive=True):
			if filename[filename.rfind("."):] in [".mid", ".midi"]:
				files.append(filename)

		np.random.shuffle(files)

		print("____ PROCESSING THE DATA")

		retMeans = np.zeros(len(quantizations))
		retStd = np.zeros(len(quantizations))
		k = 0
		for quantization in quantizations:

			trainData = data.data(quantization=quantization)
			trainData.addFiles(files[:int(train*len(files))])

			testData = data.data(quantization=quantization)
			testData.addFiles(files[int(train*len(files)):], augmentation=False)

			print(trainData.getData("length")[0])

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
		plt.xticks(np.arange(len(retMeans)), quantizations)
		plt.ylabel('Likelihood over dataset')
		plt.xlabel('Quantization')
		plt.fill_between(range(len(retMeans)), retMeans + retStd, retMeans - retStd, alpha=.5)
		plt.show()

		return (retMeans, retStd)

	def benchmarkOrder(self, folder, maxOrder, train=0.8, saveFig=False):

		np.random.seed(0)
		# We get all the midi files
		files = []
		for filename in glob(folder + '/**', recursive=True):
			if filename[filename.rfind("."):] in [".mid", ".midi"]:
				files.append(filename)

		np.random.shuffle(files)

		print("____ PROCESSING THE DATA")

		trainData = data.data()
		trainData.addFiles(files[:int(train*len(files))], augmentation=True)

		testData = data.data()
		testData.addFiles(files[int(train*len(files)):], augmentation=False)

		retMeans = np.zeros(maxOrder)
		retStd = np.zeros(maxOrder)

		print("There is", trainData.getSize(),"scores for training")

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
		plt.ylabel('Likelihood over dataset')
		plt.xlabel('Max order of the model')
		plt.fill_between(range(len(retMeans)), retMeans + retStd, retMeans - retStd, alpha=.5)
		if saveFig is False:
			plt.show()
		else:
			plt.savefig("Benchmark.eps")

		print("TRAIN DATA")
		print(files[:int(train*len(files))])

		for i in range(len(means)):
			print(files[int(train*len(files)):][i],"->",means[i])

		return (retMeans, retStd)


	def cleanWeights(self, order=None):
		"""
		Delete all trained models and fix an order if given
		"""

		if order is None:
			order = self.maxOrder

		self.LTM = []
		for viewPoint in self.viewPoints:
			self.LTM.append(longTermModel.longTermModel(viewPoint, maxOrder=self.maxOrder, use_original_PPM=self.use_original_PPM))

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
