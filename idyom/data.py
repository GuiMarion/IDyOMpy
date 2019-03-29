from idyom import score

import os
from glob import glob
import pickle
from tqdm import tqdm
import random
import numpy as np

AVAILABLE_VIEWPOINTS = ["pitch", "length"]

class data():
	"""
	Class that embed all data processing: parsing midi, representating viewpoints, ...

	:param quantization: quantization, 16 means 1/16th of beat
	:param viewpoints: Viewpoints to use, by default all are used (see data.availableViewPoints())

	:type quantization: integer
	:type viewpoints: list of string
	"""

	def __init__(self, quantization=16, viewpoints=None):

		# Dictionaries to match notes and integers
		self.itno = {}
		self.noti = {}

		# Path of the raw data
		self.folderPath = ""

		# Quantization to apply to the files
		self.quantization = quantization

		# Viewpoints to use, by default all
		self.viewpoints = viewpoints
		if self.viewpoints is None:
			self.viewpoints = AVAILABLE_VIEWPOINTS

		self.data = []

	def parse(self, path, name=None):
		"""Construct the database of tuples from an existing midi database.

		:param path: The path to the folder to load (must contain midi files).
		:param name: The name to give to the database object, optional.

		:type path: str
		:type name: str
		"""
		
		if os.path.isdir(path):
			self.path = path
			if name:
				self.name = name
			else:
				self.name = str(path)
			print()
			print("________ We are working on '" + path + "'")
			print()
		else:
			print("The path you gave is not a directory, please provide a correct directory.")
			raise RuntimeError("Invalid database directory")

		if not os.path.isdir(".TEMP"):
			os.makedirs(".TEMP")

		print("_____ Filling the database ...")
		print()

		# Number of skiped files
		skipedFiles = 0
		# Total number of files
		N = 0
		self.data = []
		for filename in glob(self.path+'/**', recursive=True):

			if filename[filename.rfind("."):] in [".mid", ".midi"]:
				if os.path.isfile(filename):
					print(" -", filename)
					try : 
						self.data.append(score.score(filename))

					except RuntimeError:
						skipedFiles += 1
					N += 1

		print()
		print("We passed a total of ", N, "files.")
		print(skipedFiles,"of them have been skiped.")
		print()

		print("_____ Augmenting database ...")
		print()

		#scores = self.augmentData(scores)

		random.shuffle(self.data)


		print("_____ Computing multiple viewpoints representation")

		self.getViewpointRepresentation()

	def getViewpointRepresentation(self):
		self.viewPointRepresentation = {}
		for viewpoint in self.viewpoints:
			self.viewPointRepresentation[viewpoint] = []
		for data in self.data:
			temp = self.dataToViewpoint(data.getData(), self.viewpoints)
			for viewpoint in self.viewpoints:
				self.viewPointRepresentation[viewpoint].append(temp[viewpoint])


	def dataToViewpoint(self, vector, viewpoints):
		"""
		Function returning the viewpoint representation of the data for a given viewpoint.
		We separate the computations for different viewpoints so it's easy to add some.
		If you want to add viewpoints you just have to change this function.

		:param vector: Vector to work with
		:param viewpoints: list of viewpoints

		:type vector: list of int, or numpy array
		:type viewpoints: list of strings

		:return: dictionnary
		"""

		representation = {}

		if "pitch" in viewpoints or "length" in viewpoints:

			pitch = []
			length = []
			new = True
			for i in range(len(vector)):
				if len(pitch) > 0 and vector[i] != pitch[-1]:
					new = True
				if new:
					pitch.append(vector[i])
					length.append(0)
					new = False
				if vector[i] == pitch[-1]:
					length[-1] += 1

			length[-1] += 2

			representation["pitch"] = pitch
			representation["length"] = length



		return representation




	def save(self, path="../DataBase/Serialized/"):
		"""Saves the database as a pickle.

		:param path: The path to the folder in which we save the file, optional.
		:type path: str
		"""

		answer = "y"

		if os.path.isfile(path+self.name+'.data'):
			print(path + self.name + ".data" + " " + " already exists, do you want to replace it ? (Y/n)")
			answer = input()

			while answer not in ["", "y", "n"]:
				print("We didn't understand, please type enter, 'y' or 'n'")
				answer = input()

			if answer in ["", "y"]:
				os.remove(path+self.name + '.data')

		if answer in ["", "y"]:
			print("____ Saving database ...")
			f = open(path+self.name + '.data', 'wb') 
			pickle.dump(self.data, f)
			f.close()

			print()
			print("The new database is saved.")
		else:
			print()
			print("We kept the old file.")


	def load(self, path):
		"""Loads  a database from a previously saved pickle.

		:param path: The path to the folder containing the data.
		:type path: str
		"""
		
		if not os.path.isfile(path):
			print("The path you entered doesn't point to a file ...")
			raise RuntimeError("Invalid file path")

		try:
			self.data = pickle.load(open(path, 'rb'))
			print("We successfully loaded the database.")
			print()
		except (RuntimeError, UnicodeDecodeError) as error:
			print("The file you provided is not valid ...")
			raise RuntimeError("Invalid file")

	def print(self):
		"""Prints the names of all items in the database."""
		
		print("_____ Printing database")
		print()
		for i in range(len(self.data)):
			print(" - ", self.data[i].name)


	def addFile(self, file):
		""" 
		Parse a midi file and return an internal representation

		:param file: file to parse
		:type file: string
		"""

		self.data.append(score.score(file))

	def getData(self, viewpoint):
		""" 
		Return data for a given viewpoint

		:param viewpoint: viewpoint (cf data.availableViewPoints())
		:type viewpoint: string
		:return: np.array((nbOfPieces, lengthMax))
		"""

		if viewpoint not in AVAILABLE_VIEWPOINTS:
			raise ValueError("We do not know this viewpoint.")
		elif viewpoint not in self.viewpoints:
			raise ValueError("We did not parse the data for this given viewpoint, try to specify it at creation of the object.")
		elif self. data == []:
			print("The data contains no items, you probably forget to parse this object.")
			return []


		return self.viewPointRepresentation[viewpoint]

	def getScore(self, viewPoint, name):
		""" 
		Return data for a given viewpoint and score

		:param viewPoint: viewpoint (cf data.availableViewPoints())
		:param name: name of the score (by default name of the file)

		:type viewPoint: string
		:type name: string

		:return: np.array(lengthMax)

		"""
		for d in self.data:
			if d.name == name:
				return d.getData()

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
		return self.getScore(viewPoint, name)[t]

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

		return AVAILABLE_VIEWPOINTS