from pypianoroll import Multitrack as proll
from pypianoroll import Track
import subprocess
import os
import numpy as np
from midi2audio import FluidSynth

try:
	import matplotlib.pyplot as plt
	plot = True
except ImportError:
	plot = False

class score:
	"""This class is used to manage midi data from the dataset.
	It uses the module pypianoroll to transform midi files in numpy arrays (piano rolls),
	and vice versa. Thus, a score object can be created from a midi
	file, but also from a numpy array representing a piano roll. 
	Score objects can be divided into snippets, plotted, transposed,
	written as new midi scores, or converted into waveForm objects.
	Attributes
	----------
	self.pianoroll : ndarray
		A numpy array representing the piano roll.
	self.name : str
		If a score class is created from a raw midi file, this attributes 
		corresponds to the name of the midi file. 
		If it is created from another score class (see extractPart
		and extractAllParts), a new name is created, that indicates wich excerpt
		of the midi file it corresponds to.
	self.velocity : bool
		True if midi velocity is considered, False otherwise.
	self.quantization : int
		MIDI quantization per beat. 24 by default.
	self.length : int
		Length of the midi data in time beats.
	self.transposition : int
		Indicates if the data is transposed from another score object. In
		semitones.
	self.outpath : str
		Folder in which temporary .wav files are stored (see toWaveForm).
	"""

	def __init__(self, pathToMidi, velocity=False, quantization=24, outPath=".TEMP/", fromArray=(None, "")):
		if fromArray[0] is not None:
			self.name = fromArray[1]
			self.pianoroll = fromArray[0]

		elif isinstance(pathToMidi, str):
			try:
				# use pypianoroll to parse the midifile
				self.pianoroll = proll(pathToMidi, beat_resolution=quantization)
				if velocity is False:
					self.pianoroll.binarize()
				self.pianoroll.remove_tracks(np.arange(len(self.pianoroll.tracks))[1:])
				self.pianoroll.trim_trailing_silence()
				self.pianoroll = self.pianoroll.get_merged_pianoroll()
				self.name = os.path.splitext(os.path.basename(pathToMidi))[0]
			except OSError:
				raise RuntimeError("incorrect midi file.")

		else:
			self.fromData(pathToMidi)
			self.name = "generation"




		# store the numpy array corresponding to the pianoroll
		self.velocity = velocity
		self.quantization = quantization

		#store length in time beat
		self.length = len(self.pianoroll)//self.quantization

		self.transposition = 0

		self.outPath = outPath

	def getPianoRoll(self):
		"""Returns the numpy array containing the pianoRoll."""
		
		return np.transpose(self.pianoroll)

	def getLength(self):
		"""Returns the length in time beats."""
		
		return self.length

	def plot(self):
		"""Plots the pianoRoll representation."""
		
		if plot == False:
			print("you cannot plot anything as matplotlib is not available")
			return

		plt.imshow(self.pianoroll.T, aspect='auto', origin='lower')
		plt.xlabel('time (beat)')
		plt.ylabel('midi note')
		plt.grid(b=True, axis='y')
		plt.yticks([0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
		   ["C-2", "C-1", "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"])
		color = plt.colorbar()
		color.set_label('velocity', rotation=270)
		plt.show()

	def extractPart(self, start, end, inBeats=False):
		"""Extract excerpt from data, and returns a score object from it.
		Parameters
		----------
		start : int
			Start position of the desired excerpt. Can be expressed in quantisized or
			unquantisized beats.
		end : int
			End position of the desired excerpt. Must be expressed in the same dimension
			as start.
		inBeats : bool
			True if start and end parameters are quantisized. False otherwise.
			
		Returns
		-------
		scorePart : score
			Extracted part of the score.
		
		"""
		
		if inBeats is True:
			if start >= 0 and end < self.length:
				pianoRollPart = self.pianoroll[start*self.quantization : end*self.quantization, : ]
				newName = self.name+ "_" + str(start) + "_" + str(end)

				scorePart = score("", fromArray=(pianoRollPart, newName))

				return scorePart
			else:
				raise IndexError("ExtractPart is asked to go over the range of the pianoRoll.")
		else:
			if start >= 0 and end < self.length*self.quantization:
				pianoRollPart = self.pianoroll[start : end, : ]
				newName = self.name+"_" + str(start) + "_" + str(end)

				scorePart = score("", fromArray=(pianoRollPart, newName))

				return scorePart
			else:
				raise IndexError("ExtractPart is asked to go over the range of the pianoRoll.")


	def extractAllParts(self, length, step=1):
		"""Extracts parts of a given length and step size, through all the data.
		Notice that it uses extractPart function for each excerpt, thus it
		creates score objects.
		Parameters
		----------
		length : int
			Length of desired excerpts in beat duration (unquantisized).
		step : int, optional
			Length of the step between each excerpts in the file. Defaults to 1.
			
		Returns
		-------
		retParts : list of score
			List of all extracted score snippets.
		"""
		
		N = self.length*self.quantization
		windowSize = length*self.quantization
		retParts = []

		for i in range((N-windowSize)//step):
			retParts.append(self.extractPart(i*step, i*step+windowSize))

		return retParts

	def toWaveForm(self, wavePath, font="SteinwayGrandPiano_1.2.sf2"):
		""" Converts the data into a waveForm object.
		It uses fluidsynth module to perform audio conversion. Notice that a
		temporary wave sound file is created, which is then immediately erased.
		SERVER is a hypercriterion that indicates if the converted data
		has to be saved into a given path.
		Parameters
		----------
		font : str, optional
			Path to a soundfont which is used by fluidsynth to create audio
			data from midi. Defaults to "SteinwayGrandPiano_1.2.sf2".
			
		Returns
		-------
		newWaveForm : waveForm
			Resulting WaveForm object.
		"""
		
		if not os.path.exists(self.outPath + "temp/"):
			os.makedirs(self.outPath + "temp/")

		midiPath = self.outPath + "temp/" +self.name + ".mid"

		pathFont = "../SoundFonts/" + font

		self.writeToMidi(midiPath)

		F = FluidSynth(pathFont)
		F.midi_to_audio(midiPath, wavePath)


		# cleaning the temporary files
		process = subprocess.Popen("rm -f " + midiPath, shell=True, stderr=subprocess.DEVNULL ,stdout=subprocess.DEVNULL)
		process.wait()

	def transpose(self, t):
		"""Returns the current pianoroll transposed by t semitones.
		Parameters
		----------
		t : int
			Value of transposition, in semitones.
			
		Returns
		-------
		result : ndarray
			Resulting transposed piano roll.
		"""
		
		result = np.empty_like(self.pianoroll)
		if t > 0:
			result[:,:t] = 0
			result[:,t:] = self.pianoroll[:,:-t]
		elif t < 0:
			result[:,t:] = 0
			result[:,:t] = self.pianoroll[:,-t:]
		else:
			result = self.pianoroll

		return result

	def getTransposed(self):
		"""Return a list of 12 transposed score objects.
		The returned list contains all the possible transpositions
		(in an octave range).
		
		Returns
		-------
		transposed_scores : list of score
			List of all transposed scores.
		"""
		
		transposed_scores = []

		# Transposes from 6 semitones down to 5 semitones up
		# And stores each transposition as a new score
		for t in range(-6, 6):
			transRoll = self.transpose(t) # transposed piano roll matrix
			newName = self.name + '_' + str(t) + "_"

			transposed_score = score("", fromArray=(transRoll, newName))
			transposed_score.transposition = t
			transposed_scores.append(transposed_score)

		return transposed_scores

	def aumgmentData(self):
		"""Function that performs data augmentation from initial data."""

		data = self.getTransposed()

		return data


	def writeToMidi(self, midiPath):
		"""Saves the piano roll data in a midi file.
		Parameters
		----------
		midiPath : str
			Folder in which the corresponding midi data is saved.
		"""
		
		tempTrack = Track(pianoroll=self.pianoroll, program=0, is_drum=False,
									name=self.name)
		tempMulti = proll(tracks=(tempTrack,), beat_resolution=self.quantization)
		tempMulti.write(midiPath)		

	def getData(self):
		"""
		Returns the data as a sequence of notes without silences
		"""
		P = self.getPianoRoll()
		ret = np.zeros(len(P[0])).astype(int) - 1
		for i in range(len(P[0])):
			for j in range(len(P)):
				if P[j][i] != 0:
					ret[i] = j

		# We delete the silences at the end
		k = 0
		for i in range(1, len(ret)):
			if ret[-i] == -1:
				k += 1
			else:
				if k > 0:
					ret = ret[:-1]
				break


		# in order to delete silences at the bigining
		if ret[0] == -1:
			start = True
		else:
			start = False

		# delete silences at the bigining and replace silences in the piece by at longer duration of note (time onset)
		start_index = 0
		for i in range(1, len(ret)):
			if ret[i] == -1 and ret[i-1] != -1:
				ret[i] = ret[i-1]
				start = False
			elif start:
				start_index += 1

		ret = ret[start_index:]

		# for i in range(1, len(ret)-2):
		# 	if ret[i] == -1 and ret[i-1] != -1 and ret[i+1] != -1:
		# 		ret[i] = ret[i-1]
		# 	if ret[i] == -1 and ret[i-1] != -1 and ret[i+2] != -1:
		# 		ret[i] = ret[i-1]
		
		return ret

	def fromData(self, data):
		"""
		Construct the score object from a sequence of notes.
		"""

		P = np.zeros((len(data), 128))
		for i in range(len(data)):
			if data[i] >= 0:
				P[i][int(data[i])] = 80

		self.pianoroll = P