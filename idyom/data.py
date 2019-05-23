from idyom import score

import os
from glob import glob
import pickle
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

AVAILABLE_VIEWPOINTS = ["pitch", "length"]

class data():
	"""
	Class that embed all data processing: parsing midi, representating viewpoints, ...

	:param quantization: quantization, 16 means 1/16th of beat
	:param viewpoints: Viewpoints to use, by default all are used (see data.availableViewPoints())

	:type quantization: integer
	:type viewpoints: list of string
	"""

	def __init__(self, quantization=24, viewpoints=None, deleteDuplicates=True):

		# Dictionaries to match notes and integers
		self.itno = {}
		self.noti = {}

		# Path of the raw data
		self.folderPath = ""

		# Quantization to apply to the files
		self.quantization = quantization

		# True if we allow the program to delete duplicates
		self.deleteDuplicates = deleteDuplicates

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
		self.files = []
		for filename in glob(self.path+'/**', recursive=True):

			if filename[filename.rfind("."):] in [".mid", ".midi"]:
				if os.path.isfile(filename):
					print(" -", filename)
					try : 
						self.data.append(score.score(filename))
						self.files.append(filename)

					except RuntimeError:
						skipedFiles += 1
					N += 1

		print()
		print("We passed a total of ", N, "files.")
		print(skipedFiles,"of them have been skiped.")
		print()

		print("_____ Computing multiple viewpoints representation")

		self.getViewpointRepresentation()

		print("_____ Augmenting database ...")
		print()

		self.augmentData()

		#random.shuffle(self.data)

		print("Data processing done.")

	def getViewpointRepresentation(self):
		self.viewPointRepresentation = {}
		for viewpoint in self.viewpoints:
			self.viewPointRepresentation[viewpoint] = []
		for data in self.data:
			temp = self.dataToViewpoint(data.getData(), self.viewpoints)
			if self.deleteDuplicates and temp["pitch"] in self.viewPointRepresentation["pitch"]:
				print("We found a duplicate, we ignore it. We encourage you to check your dataset with App.py -c 1")
			else:
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

			pitch = [vector[0]]
			length = [1]
			for i in range(1, len(vector)):
				if len(pitch) > 0 and vector[i] != pitch[-1] and vector[i] != -1:
					pitch.append(vector[i])
					length.append(1)

				elif len(pitch) > 0 and vector[i-1] == -1:
					length[-1] += 1
					pitch.append(vector[i])
					length.append(1)

				elif vector[i] == pitch[-1]:
					length[-1] += 1

			length[-1] += 2

			length = self.quantize(length)

			representation["pitch"] = pitch
			representation["length"] = np.concatenate((np.array([0]), length[:-1]), axis=None) # We shift the length so it's really a time onset from the previous note



		return representation

	def quantize(self, length):

		length = np.round(np.array(length) / (24/self.quantization)) * 24/self.quantization

		return length.astype(int)

	def augmentData(self):
		"""
		Augments the data with some techniques like transposition.
		
		"""
		self.augmentByTransposition()

	def augmentByTransposition(self):

		augmented = []
		for elem in self.viewPointRepresentation["pitch"]:
			for t in range(-6,6):
				augmented.append(np.array(elem) + t)

		self.viewPointRepresentation["pitch"].extend(augmented)

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
		Parse a midi file

		:param file: file to parse
		:type file: string
		"""

		self.data.append(score.score(file))

		self.getViewpointRepresentation()


	def addFiles(self, files, augmentation=True):
		""" 
		Parse a list of midi file

		:param files: files to parse
		:type file: list of string
		"""
		for file in files:
			print("__", file)
			self.data.append(score.score(file))

		print("___ Constructing viewPoint representation")

		self.getViewpointRepresentation()

		if augmentation:
			print("_____ Augmenting database ...")
			print()

			self.augmentData()

	def addScore(self, s):
		""" 
		Parse a midi file and return an internal representation

		:param file: file to parse
		:type file: string
		"""


		if isinstance(s, score.score):
			self.data.append(s)
			self.getViewpointRepresentation()
		else:
			print("This object you gave is not a score object, we cannot import it.")

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
		elif self.data == []:
			print("The data contains no items, you probably forget to parse this object.")
			return []


		return self.viewPointRepresentation[viewpoint]

	def plotScores(self):
		dat = []
		k = 0
		for viewpoint in self.viewPointRepresentation:
			dat.append([])
			for score in self.viewPointRepresentation[viewpoint]:
				if viewpoint is "length":
					dat[k].append(np.mean(score))
				elif viewpoint is "pitch":
					dat[k].append(np.mean(np.diff(score)))

			k += 1

		plt.scatter(dat[0][:len(dat[1])],dat[1])

		plt.title('Database')
		plt.xlabel('Average 1-note interval')
		plt.ylabel('Average note onset')

		plt.show()

	def getScoresFeatures(self):

		dat = []
		k = 0
		for viewpoint in self.viewPointRepresentation:
			dat.append([])
			for score in self.viewPointRepresentation[viewpoint]:
				if viewpoint is "length":
					dat[k].append(np.mean(score))
				elif viewpoint is "pitch":
					dat[k].append(np.mean(np.diff(score)))

			k += 1

		return dat, self.files
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

	def getSizeofPiece(self, piece):
		"""
		Returns the size of a given piece from its index
		:param piece: index of a piece
		
		:type piece: int
		:return: length of the piece (int)
		"""

		return len(self.viewPointRepresentation[AVAILABLE_VIEWPOINTS[0]][piece])

	def getSize(self):
		"""
		Returns the number of exemples
		"""

		return len(self.getData(AVAILABLE_VIEWPOINTS[0]))