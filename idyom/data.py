class data():
	"""
	Class that embed all data processing: parsing midi, representating viewpoints, ...

	:param quantization: quantization, 16 means 1/16th of beat
	:type quantization: integer
	"""

	def __init__(self, quantization=16):

		# Dictionaries to match notes and integers
		self.itno = {}
		self.noti = {}

		# Path of the raw data
		self.folderPath = ""

		# Quantization to apply to the files
		self.quantization = quantization

	def load(self, file):
		"""
		Load pre-processed data from file.

		:param file: file containing the data
		:type file: string
		"""

		return 0

	def save(self, path):
		""" 
		Save the processed data into a file

		:param path: path to write the data to
		:type path: string
		"""
		return 0

	def parse(self, dataFolder):
		""" 
		Parse a folder of file 


		:param dataFolder: folder containing the files
		:type dataFolder: string
		"""
		return 0

	def parseFile(self, file):
		""" 
		Parse a midi file and return an internal representation

		:param file: file to parse
		:type file: string
		"""
		return 0

	def getData(self, viewPoint):
		""" 
		Return data for a given viewpoint

		:param viewPoint: viewpoint (cf data.availableViewPoints())
		:type viewPoint: string
		:return: np.array((nbOfPieces, lengthMax))
		"""
		return 0

	def getScore(self, viewPoint, name):
		""" 
		Return data for a given viewpoint and score

		:param viewPoint: viewpoint (cf data.availableViewPoints())
		:param name: name of the score (by default name of the file)

		:type viewPoint: string
		:type name: string

		:return: np.array(lengthMax)

		"""
		return 0

	def getNote(self, viewPoint, name, t):
		""" 
		Return data for a given viewpoint, score and index

		:param viewPoint: viewpoint (cf data.availableViewPoints())
		:param name: name of the score (by default name of the file)
		:param t: index

		:type viewPoint: string
		:type name: string
		:type t: integer
		:return: integer corresponding to the note (cf. data.intToNote())
		"""
		return 0

	def noteToInt(self, note):
		""" 
		Return integer of a note from its name

		:param note: note name

		:type note: string
		:return: integer corresponding to the note
		"""

		if note in self.noti:
			return self.noti[note]
		else:
			print("This note is not known ...")
			return None

	def intToNote(self, note):
		""" 
		Return the name of a note from its integer

		:param note: note integer
		
		:type note: int
		:return: string name corresponding to the note
		"""

		if note in self.itno:
			return self.itno[note]
		else:
			print("This note is not known ...")
			return None


	def availableViewPoints(self):
		""" 
		Return the list of available viewPoints

		:return: list of strings
		"""

		return []