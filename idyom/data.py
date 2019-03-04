
class data():
	def __init__(self, folder, quantization=16):

		# Dictionary to match notes and integers
		self.intToNote = {}
		self.noteToInt = {}

		# Path of the raw data
		self.folderPath = ""

		# Quantization to apply to the files
		self.quantization = quantization

	def load(self, file):
		# load the file contening already processed data

		return 0

	def save(self, path):
		# save the processed data into a file

		return 0

	def parse(self, dataFolder):
		# parse a folder of file 

		return 0

	def parseFile(self, file):
		# parse a midi file and return an internal representation

		return 0

	def getData(self):
		# return the data

		return 0

	def getScore(self, name):
		# return a given score

		return 0

	def getNote(self, name, t):
		# return a given note of a given score

		return 0