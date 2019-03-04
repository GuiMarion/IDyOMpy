import data
import markovChain
import longTermModel

class idyom():
	def __init__(self, maxOrder=None, viewPoints=["pitch", "length"], dataTrain=None, dataTrial=None):

		# viewpoints to use for the model
		self.viewPoints = viewPoints

		# list of all models for each viewpoints
		self.LTM = []
		for v in viewPoints:
			self.LTM.append(longTermModel.longTermModel(maxOrder))

		# data to train with
		self.dataTrain = data.data(dataTrain)

		# data for the trial
		self.dataTrial = data.data(dataTrial)

	def train(self, data):

		# train the models from data
		k = 0
		for viewPoint in viewPoints:
			self.LTM[k].train(data.getData(viewPoint))

	def predict(self, sequence):
		# return the probability ditribution given a sequence

		return 0

	def getLikelihood(self, sequence, note):
		# return the likelihood of a note given a sequence

		return 0

	def getLikelihoodfromData(self, data):
		# return likelihood over a all dataset

		return 0

	def generate(self, length):
		# return a piece of music 

		return 0





