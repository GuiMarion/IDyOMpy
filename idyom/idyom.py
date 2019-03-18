from idyom import data
from idyom import markovChain
from idyom import longTermModel

class idyom():
	"""
	This module represent the entire model, this is what you want to interact with if you only want to use the model.

	:param maxOrder: maximal order of the model
	:param viewPoints: viewPoint to use, cf. data.getViewPoints()

	:type maxOrder: int
	:type viewPoints: list of strings
	"""
	def __init__(self, maxOrder=None, viewPoints=["pitch"], dataTrain=None, dataTrial=None):

		# viewpoints to use for the model
		self.viewPoints = viewPoints

		# list of all models for each viewpoints
		self.LTM = []
		for viewPoint in viewPoints:
			self.LTM.append(longTermModel.longTermModel(viewPoint, maxOrder))

	def train(self, data):
		"""
		Train the models from data
		
		:param data: data to train from

		:type data: class data or list of int
		"""

		k = 0
		for viewPoint in viewPoints:
			self.LTM[k].train(data.getData(viewPoint))

	def predict(self, sequence):
		"""
		Return the probability ditribution given a sequence
		
		:param sequence: a sequence of viewPoint data, cf. data.getData(viewPoint)

		:type sequence: np.array(length)

		:return: dictionary, dico[note] = probability
		"""

		return 0

	def getLikelihood(self, sequence, note):
		"""
		Return the likelihood of a note given a sequence
		
		:param sequence: a sequence of viewPoint data, cf. data.getData(viewPoint)
		:param note: integer of name of the note

		:type sequence: np.array(length)
		:type note: int or string

		:return: float value of the likelihood
		"""

		return 0

	def getLikelihoodfromData(self, data):
		"""
		Return likelihood over a all dataset
		
		:param data: data to process into

		:type data: class data

		:return: np.array((nbOfPiece, maximalLength))

		"""
		return 0

	def generate(self, length):
		"""
		Return a piece of music generated using the model.

		:param length: length of the output

		:type length: int

		:return: class piece
		"""

		return 0





