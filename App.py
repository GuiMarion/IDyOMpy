"""
Enter point of the program.
"""
from idyom import idyom
from idyom import data
from lisp import parser as lisp
from idyom import jumpModel

from optparse import OptionParser
from glob import glob
from tqdm import tqdm
import unittest
import matplotlib.pyplot as plt
import numpy as np
import os

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

def replaceinFile(file, tochange, out):
	s = open(file).read()
	s = s.replace(tochange, out)
	f = open(file, "w")
	f.write(s)
	f.close()

def cross_validation(folder, k_fold=10, maxOrder=20, quantization=24, jump=False):
	"""

	"""

	np.random.seed(0)

	Likelihoods = []

	files = []
	for filename in glob(folder + '/**', recursive=True):
		if filename[filename.rfind("."):] in [".mid", ".midi"]:
			files.append(filename)

	np.random.shuffle(files)

	k_fold = len(files) // int(k_fold)

	for i in range(len(files)//k_fold):
		trainData = files[:i*k_fold] + files[(i+1)*k_fold:]
		evalData = files[i*k_fold:(i+1)*k_fold]

		# Our IDyOM
		L = idyom.idyom(maxOrder=maxOrder, jump=jump)
		M = data.data(quantization=quantization)
		M.addFiles(trainData)
		L.train(M)

		for file in evalData:
			Likelihoods.append(np.mean(L.getLikelihoodfromFile(file)))

	return Likelihoods

def compareJump(folder, k_fold=2):
	"""
	Compare the likelihood between idyom model and jump model.
	"""

	likelihood1 = cross_validation(folder, k_fold=k_fold, jump=False)
	likelihood2 = cross_validation(folder, k_fold=k_fold, jump=True)

	plt.ylabel("Likelihood")
	plt.bar([0, 1], [np.mean(likelihood1), np.mean(likelihood2)], color="b", yerr=[1.96*np.std(likelihood1)/np.sqrt(len(likelihood1)), 1.96*np.std(likelihood2)/np.sqrt(len(likelihood2))])
	plt.show()

def compareJump(folder, k_fold=2):
	"""
	Compare the likelihood between idyom model and jump model.
	"""

	likelihood1 = cross_validation(folder, k_fold=k_fold, jump=False)
	likelihood2 = cross_validation(folder, k_fold=k_fold, jump=True)

	plt.ylabel("Likelihood")
	plt.bar([0, 1], [np.mean(likelihood1), np.mean(likelihood2)], color="b", yerr=[1.96*np.std(likelihood1)/np.sqrt(len(likelihood1)), 1.96*np.std(likelihood2)/np.sqrt(len(likelihood2))])
	plt.show()

def plotLikelihood(folder, k_fold=2):
	"""
	Compare the likelihood between idyom model and jump model.
	"""

	likelihood1 = cross_validation(folder, k_fold=k_fold, jump=False)

	plt.ylabel("Likelihood")
	plt.bar([0], [np.mean(likelihood1)], color="b", yerr=[np.std(likelihood1)])
	plt.show()

	print()
	print()
	print()

	print("Mean:", np.mean(likelihood1))
	print("Std:", np.std(likelihood1))


def compareWithLISP(folder):
	"""
	Start comparisons between our idyom and the one in lisp.
	This function, will add the dataset to lisp, and start training.
	You should have lisp and idyom already installed.
	"""

	if not os.path.exists("lisp/midis/"):
	    os.makedirs("lisp/midis/")

	os.system("rm -rf lisp/midis/*")

	# Add folder to lisp database

	replaceinFile("lisp/compute.lisp", "FOLDER", folder)

	# Compute with LISP IDyOM

	os.system("sbcl --noinform --load lisp/compute.lisp")

	replaceinFile("lisp/compute.lisp", folder, "FOLDER")


	folder = "lisp/midis/"

	# Our IDyOM

	likelihoods1 = cross_validation(folder, maxOrder=20, quantization=6, k_fold=10)


	# LISP version

	L2 = lisp.getDico("lisp/12-cpitch_onset-cpitch_onset-nil-nil-melody-nil-10-both+-nil-t-nil-c-nil-t-t-x-3.dat")

	likelihood2 = lisp.getLikelihood(L2)

	plt.ylabel("Likelihood")
	plt.bar([0, 1], [np.mean(likelihoods1), likelihood2[0]], color="b", yerr=[1.96*np.std(likelihoods1)/np.sqrt(len(likelihoods1)), 1.96*likelihood2[1]/np.sqrt(likelihood2[2])])
	plt.show()


	# LATER 
	quit()
	plt.ylabel("Likelihood")
	plt.xlabel("time")
	plt.plot(L2['1']["probability"])
	plt.plot(L.getLikelihoodfromFile(folder+L2['1']["melody.name"][0][1:-1] + ".mid"))
	plt.show()





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

	parser.add_option("-l", "--lisp", type="string",
					  help="plot comparison with the lisp version",
					  dest="lisp", default="")

	parser.add_option("-j", "--jump", type="string",
					  help="plot comparison with the jump",
					  dest="jump", default="")

	parser.add_option("-p", "--plot", type="string",
					  help="plot likelihood of idyom model",
					  dest="plot", default="")

	parser.add_option("-k", "--k_fold", type="int",
				  help="set the value of k for cross validation",
				  dest="k", default=None)

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

	if options.jump != "":		
		compareJump(options.jump)

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

	if options.lisp != "":		
		compareWithLISP(options.lisp)

	if options.plot != "":
		if options.k is None:		
			plotLikelihood(options.plot)
		else:
			plotLikelihood(options.plot, k_fold=options.k)

