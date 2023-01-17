import mido
import collections
import numpy as np
try:
	import matplotlib.pyplot as plt
	plot = True
except ImportError:
	plot = False

def getNotesFromMidi(midiFile):
	"""
	Compute a list of Note objects from a midi file.
	----------
	midiFile : str
		Filepath of the midi file.

	Returns
	-------
	notes : list of Note.
		List of Note objects.
	"""

	mido_obj = mido.MidiFile(midiFile)

	# Delta to cumulative
	for track in mido_obj.tracks:
		tick = int(0)
		for event in track:
			event.time += tick
			tick = event.time
	NOTES = []
	# We extract the notes from the events
	for track_idx, track in enumerate(mido_obj.tracks):
		# Keep track of last note on location:
		last_note_on = collections.defaultdict(list)

		notes = []
		for event in track:
			if event.type == 'note_on' and event.velocity > 0:
				# Store this as the last note-on location
				note_on_index = (event.channel, event.note)
				last_note_on[note_on_index].append((
					event.time, event.velocity))
			# Note offs can also be note on events with 0 velocity
			elif event.type == 'note_off' or (event.type == 'note_on' and
											  event.velocity == 0):
				# Check that a note-on exists (ignore spurious note-offs)
				key = (event.channel, event.note)
				if key in last_note_on:
					# Get the start/stop times and velocity of every note
					# which was turned on with this instrument/drum/pitch.
					# One note-off may close multiple note-on events from
					# previous ticks. In case there's a note-off and then
					# note-on at the same tick we keep the open note from
					# this tick.
					end_tick = event.time
					open_notes = last_note_on[key]

					notes_to_close = [
						(start_tick, velocity)
						for start_tick, velocity in open_notes
						if start_tick != end_tick]
					notes_to_keep = [
						(start_tick, velocity)
						for start_tick, velocity in open_notes
						if start_tick == end_tick]

					for start_tick, velocity in notes_to_close:
						start_time = start_tick
						end_time = end_tick
						# Create the note event
						note = Note(velocity, event.note, start_time,
									end_time)
			
						notes.append(note)

					if len(notes_to_close) > 0 and len(notes_to_keep) > 0:
						# Note-on on the same tick but we already closed
						# some previous notes -> it will continue, keep it.
						last_note_on[key] = notes_to_keep
					else:
						# Remove the last note on for this instrument
						del last_note_on[key]

			elif event.type == 'end_of_track':
				# This is not more reliable than looking at the encoded duration.
				pass
		NOTES.append(notes)

	isPoly = False
	for N in NOTES:
		if len(N) > 0:
			if isPoly:
				raise RuntimeError("This file is polyphonic, I cannot handle this: "+str(midiFile))
			notes = N
			isPoly = True

	if not isPoly:
		raise RuntimeError("This file is empty, I cannnot handle this ... " + str(midiFile))

	return notes, mido_obj.ticks_per_beat

def getRepresentationFromNotes(notes, quantization=24):
	"""
	Get a list of notes and return the viewpoint (pitch, durations) representation
	----------
	notes : list of Notes
		List of Note objects.
	quantization : int (optional)
		Quantization factor, it is the number of ticks in 1 beat. Default is 24, I recommand this as it is divisible by 2 and 3 without being too large.
		
	Returns
	-------
	pitch : list of int
		List of the midi pitch cotained in the file.
	duration : list of int
		List of the quantized (in ticks not time) durations of the notes
	velocity : list of int
		List of velocities (intensity).
	silenceBegining : int
		The number of ticks of silence at the beging of the file (if anacrouse).
	"""
	notes, ticks_per_beat = notes

	pitch = []
	duration = []
	velocity = []

	for i in range(len(notes)-1):
		if notes[i+1].start < notes[i].end:
			raise RuntimeError("This Midi is polyphonic, I cannot handle this.")

		#noteDuration = notes[i].get_duration()
		# This one is more reliable 
		durationUntilNextNote = notes[i+1].start - notes[i].start
		pitch.append(notes[i].pitch)
		duration.append(round(quantization*durationUntilNextNote/ticks_per_beat))
		velocity.append(notes[i].velocity)

	pitch.append(notes[-1].pitch)
	velocity.append(notes[-1].velocity)
	# This one is not reliable but IDyOM removes the last duration anyway
	duration.append(round(quantization*(notes[-1].end-notes[-1].start)/ticks_per_beat))

	silenceBegining = round(quantization*notes[0].start/ticks_per_beat)

	return pitch, duration, velocity, silenceBegining

def readMidi(file,quantization=24):
	"""
	Open the passed midi file and returns the viewpoint (pitch, duration) representation.
	Parameters
	----------
	file : str
		Path to the midi file. It needs to be monophonic.
	quantization : int (optional)
		Quantization factor, it is the number of ticks in 1 beat. Default is 24, I recommand this as it is divisible by 2 and 3 without being too large.
		
	Returns
	-------
	pitch : list of int
		List of the midi pitch cotained in the file.
	duration : list of int
		List of the quantized (in ticks not time) durations of the notes.
	velocity : list of int
		List of velocities (intensity).
	silenceBegining : int
		The number of ticks of silence at the beging of the file (if anacrouse).
	"""
	return getRepresentationFromNotes(getNotesFromMidi(file))


class Note(object):
    """A simple Note object.
    Parameters
    ----------
    velocity : int
        Note velocity.
    pitch : int
        Note pitch, as a MIDI note number.
    start : float
        Note on time, absolute, in ticks.
    end : float
        Note off time, absolute, in ticks.
    """

    def __init__(self, velocity, pitch, start, end):
        self.velocity = velocity
        self.pitch = pitch
        self.start = start
        self.end = end

    def get_duration(self):
        """Get the duration of the note in ticks."""
        return self.end - self.start

    def __repr__(self):
        return 'Note(start={:d}, end={:d}, pitch={}, velocity={})'.format(
            self.start, self.end, self.pitch, self.velocity)

class Score():
	"""This class is used to manage midi data from the dataset.
	It uses the module mido to transform midi files in numpy arrays. 
	Thus, a score object can be created from a midi file.
	Attributes
	----------
	name : str
		Name of the original midi file.
	pitch : list of int
		List of the midi pitches.
	duration : list of int
		List of the durations of the notes.
	velocity: List of int
	List of the velocities of the notes (intensity).
	silenceBegining : int
		Number of ticks of silence at the bigining of the file.
	quantization : int
		Ticks quantization per beat. 24 by default.
	"""

	def __init__(self, fileName, quantization=24):
		self.name = fileName
		self.quantization = quantization
		self.pitch, self.duration, self.velocity, self.silenceBegining = readMidi(fileName)


	def getLength(self):
		"""Returns the length in time beats."""
		
		return (np.sum(self.duration)+self.silenceBegining)

	def getData(self):
		"""
		Returns the data as a list of pitch and a list of durations
		"""
		return self.pitch, self.duration

	def getOnsets(self, sr=64, tempo=120):
		"""
		Returns the onsets representation, useful for EEG analysis for instance.
		----------
		sr : int
			Sampling rate for the returned signal (64 by default).
		tempo : int
			Tempo you want to render the signal at (default 120.

		Returns
		-------
		onsets : list of int
			Signal sampled at sr Hz containing 0s and 1s. Ones means that there is a note.
		"""
		onsets = []
		onsets.extend([0]*round(self.silenceBegining*sr*60/tempo))
		for d in self.duration:
			onsets.append(1)
			onsets.extend([0]*(round(d*sr*60/tempo)-1))

		return onsets

	def getPianoRoll(self):
		"""
		Returns the numpy array containing the pianoRoll. The pitch is render in a 128 dimensions axis.
		
		Returns
		-------
		pianoRoll : numpy array
			Numpy array containing the pianoroll the pitch dimension shape is always 128.
		"""

		mat = np.zeros((128, int(self.getLength())))

		tick = self.silenceBegining
		for i in range(len(self.pitch)):
			mat[self.pitch[i], tick:tick+self.duration[i]] = self.velocity[i]
			tick += self.duration[i]

		return mat


	def plot(self):
		"""Plots the pianoRoll representation."""
		
		if plot == False:
			print("you cannot plot anything as matplotlib is not available")
			return
		minPitch = min(self.pitch)
		maxPitch = max(self.pitch)
		pitchRange = maxPitch - minPitch

		mat = np.zeros((int(pitchRange)+1, int(self.getLength())))

		tick = self.silenceBegining
		for i in range(len(self.pitch)):
			mat[self.pitch[i]-minPitch, tick:tick+self.duration[i]] = self.velocity[i]
			tick += self.duration[i]

		plt.imshow(mat, aspect='auto', origin='lower')
		plt.xlabel('time (beat)')
		plt.ylabel('midi note')
		plt.grid(b=True, axis='y')
		plt.yticks(list(range(maxPitch-minPitch)), list(range(minPitch, maxPitch)))
		color = plt.colorbar()
		color.set_label('velocity', rotation=270)
		plt.show()

