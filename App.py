"""
Enter point of the program.
"""
from idyom import idyom
from idyom import data

from optparse import OptionParser
from glob import glob
from tqdm import tqdm
import unittest
import matplotlib.pyplot as plt

def comparePitches(list1, list2, k=0.9):
	"""
	Compare two list of pitches, with a criterion k
	"""
	score = 0

	for i in range(min(len(list1), len(list2))):
		score += list1[i] == list2[i]

	if score > int(k*min(len(list1), len(list2))):
		return True
	else:
		return False

def checkDataSet(folder):
	"""
	Function that check if the dataset is corrupted (contains duplicates).
	"""

	files = []
	for filename in glob(folder + '/**', recursive=True):
		if filename[filename.rfind("."):] in [".mid", ".midi"]:
			files.append(filename)

	D = data.data(deleteDuplicates=False)
	D.addFiles(files)
	DATA = D.getData("pitch")

	delete = []
	delete_pitches = []

	for i in range(len(files)):
		for j in range(i, len(files)):
			if i != j and comparePitches(DATA[i], DATA[j]):

				print(files[i], "matches", files[j])


				# We recommand to delete the smallest one
				if len(DATA[i]) > len(DATA[j]):
					for d in delete_pitches:
						if comparePitches(d, DATA[i]):
							delete.append(files[i])
							delete_pitches.append(DATA[i])
							break

					delete.append(files[j])
					delete_pitches.append(DATA[j])
				else:
					for d in delete_pitches:
						if comparePitches(d, DATA[j]):
							delete.append(files[j])
							delete_pitches.append(DATA[j])
							break

					delete.append(files[i])
					delete_pitches.append(DATA[i])			

	if len(delete) > 0:
		print("We recommand you to delete the following files because they are duplicates:")
		print(list(set(delete)))
	else:
		print("We did not find any duplicates.")


def main():
	"""
	Call this method to easily use the program.
	"""

	pass

if __name__ == "__main__":

	usage = "usage %prog [options]"
	parser = OptionParser(usage)

	parser.add_option("-t", "--test", type="int",
					  help="1 if you want to launch unittests",
					  dest="tests", default=0)

	parser.add_option("-o", "--opti", type="string",
					  help="launch optimisation of hyper parameters on the passed dataset",
					  dest="hyper", default="")

	parser.add_option("-c", "--check", type="string",
					  help="check the passed dataset",
					  dest="check", default="")

	parser.add_option("-g", "--generate", type="int",
					  help="generate piece of the passed length",
					  dest="generate", default=0)

	parser.add_option("-s", "--surprise", type="string",
					  help="return the surprise over a given dataset",
					  dest="surprise", default="")

	options, arguments = parser.parse_args()


	if options.tests == 1:
		loader = unittest.TestLoader()

		start_dir = "unittests/"
		suite = loader.discover(start_dir)

		runner = unittest.TextTestRunner()
		runner.run(suite)

	if options.hyper != "":
		L = idyom.idyom(maxOrder=30)

		L.benchmarkQuantization(options.hyper,train=0.8)
		L.benchmarkOrder(options.hyper, 24, train=0.8)

	if options.check != "":
		checkDataSet(options.check)

	if options.generate != 0:		
		L = idyom.idyom(maxOrder=30)
		M = data.data(quantization=6)
		#M.parse("../dataset/")
		M.parse("dataset/")
		L.train(M)
		s = L.generate(int(options.generate))
		s.plot()
		s.writeToMidi("exGen.mid")

	if options.surprise != "":
		L = idyom.idyom(maxOrder=30)
		M = data.data(quantization=6)
		#M.parse("../dataset/")
		M.parse("dataset/")
		L.train(M)

		S = L.getSurprisefromFile(options.surprise, zero_padding=True)

		plt.plot(S)
		plt.xlabel("Time in quantization step")
		plt.ylabel("Expected surprise (-log2(p))")
		plt.show()

		print(S)



