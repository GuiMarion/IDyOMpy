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
import pickle
import time
import scipy.io as sio
import math

SERVER = True

if SERVER:
	plt.ioff()


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

	validationFiles = []

	for i in tqdm(range(math.ceil(len(files)/k_fold))):
		trainData = files[:i*k_fold] + files[(i+1)*k_fold:]
		evalData = files[i*k_fold:(i+1)*k_fold]

		# Our IDyOM
		L = idyom.idyom(maxOrder=maxOrder, jump=jump)
		M = data.data(quantization=quantization)
		M.addFiles(trainData)

		L.train(M)

		for file in evalData:
			tmp = L.getLikelihoodfromFile(file)
			for i in range(len(tmp)):
				if tmp[i] != tmp[i]:
					tmp[i] = 1/30
			Likelihoods.append(np.mean(tmp))
			filename = file[file.rfind("/")+1:file.rfind(".")]
			validationFiles.append(filename)

	return Likelihoods, validationFiles


def compareLikelihoods(x1, x2, name="kikou.eps"):

	plt.title("Likelihoods over pieces")
	plt.xlabel("pieces")
	plt.ylabel("likelihood")
	ax = plt.subplot(111)

	for i in range(len(x1)):
		ax.bar(i-0.2, x1[i], width=0.2, color='b', align='center')
		ax.bar(i, x2[i], width=0.2, color='g', align='center')

	if not SERVER:
		plt.show()
	else:
		plt.savefig("figs/server/"+name+"_1.eps")
		plt.close()

	plt.title("Likelihood diferences over pieces")
	plt.xlabel("pieces")
	plt.ylabel("likelihood diference (idyom - jump)")
	plt.plot(np.array(x1)-np.array(x2))
	plt.plot(np.zeros(len(x1)))

	if not SERVER:
		plt.show()
	else:
		plt.savefig("figs/server/"+name+"_2.eps")
		plt.close()

def compareJump(folder, k_fold=10):
	"""
	Compare the likelihood between idyom model and jump model.
	"""
	# if os.path.isfile(".IDyOM.save"):
	# 	likelihood1, files1 = pickle.load(open(".IDyOM.save", 'rb'))
	# 	print("We loaded idyom model from pickle.")
	# else:
	# 	print("We store idyom model for later.")
	# 	likelihood1, files1 = cross_validation(folder, k_fold=k_fold, jump=False)
	# 	pickle.dump((likelihood1, files1), open(".IDyOM.save", 'wb'))
	

	likelihood1, files1 = cross_validation(folder, k_fold=k_fold, jump=False)
	likelihood2, files2 = cross_validation(folder, k_fold=k_fold, jump=True)

	plt.title("IDyOMpy - JUMP")
	plt.ylabel("Likelihood")
	plt.bar([0, 1], [np.mean(likelihood1), np.mean(likelihood2)], color="b", yerr=[1.96*np.std(likelihood1)/np.sqrt(len(likelihood1)), 1.96*np.std(likelihood2)/np.sqrt(len(likelihood2))])
	
	if not SERVER:
		plt.show()
	else:
		plt.savefig("figs/server/Jump/JUMPCompare.eps")
		plt.close()

	print("IDyOM")
	print("Mean:", np.mean(likelihood1))
	print("Std:", np.std(likelihood1))

	print("JUMP")
	print("Mean:", np.mean(likelihood2))
	print("Std:", np.std(likelihood2))

	M = data.data()
	M.parse(folder)
	dat1, files3 = M.getScoresFeatures()

	dico = dict(zip(files1, likelihood1))

	dico2 = dict(zip(files2, likelihood2))

	x1 = []
	x2 = []

	for file in files1:
		if file in dico2 and dico[file] is not None and dico2[file] is not None:
			x1.append(dico[file])
			x2.append(dico2[file])

	compareLikelihoods(x1, x2, name="Jump/compareLikelihoods")


	M = data.data()
	M.parse(folder, augment=False)

	dat2, files4 = M.getScoresFeatures()

	weights = []
	colors = []

	for file in range(len(x1)):
		weights.append(80000*abs(x1[file]-x2[file])**2)
		if x1[file]-x2[file] < 0:
			colors.append('coral')
		elif x1[file]-x2[file] > 0:
			colors.append('deepskyblue')
		else:
			colors.append('black')


	plt.scatter(dat2[0][:len(dat2[1])],dat2[1], s=weights, c=colors)

	plt.title('IDyOMpy - Jump')
	plt.xlabel('Average 1-note interval')
	plt.ylabel('Average note onset')

	if not SERVER:
		plt.show()
	else:
		plt.savefig("figs/server/Jump/scoreSpaceIDyOMpy_VS_Jump.eps")
		plt.close()

def plotLikelihood(folder, k_fold=2):
	"""
	Compare the likelihood between idyom model and jump model.
	"""

	likelihood1, files = cross_validation(folder, k_fold=k_fold, jump=True)

	print(likelihood1)
	print(files)

	plt.ylabel("Likelihood")
	plt.bar([0], [np.mean(likelihood1)], color="b", yerr=[np.std(likelihood1)])
	plt.show()

	print()
	print()
	print()

	print("Mean:", np.mean(likelihood1))
	print("Std:", np.std(likelihood1))

	M = data.data()
	M.parse(folder)
	dat, files2 = M.getScoresFeatures()

	dico = dict(zip(files, likelihood1))

	weights = []

	for file in files2:
		if file in dico:
			weights.append(500*dico[file]**2)
		else:
			weights.append(0)


	plt.scatter(dat[0][:len(dat[1])],dat[1], s=weights)

	plt.title('Database')
	plt.xlabel('Average 1-note interval')
	plt.ylabel('Average note onset')

	plt.show()

def compareWithLISP(folder):
	"""
	Start comparisons between our idyom and the one in lisp.
	This function, will add the dataset to lisp, and start training.
	You should have lisp and idyom already installed.
	"""

	# if not os.path.exists("lisp/midis/"):
	# 	os.makedirs("lisp/midis/")

	# os.system("rm -rf lisp/midis/*")

	# # Add folder to lisp database

	# replaceinFile("lisp/compute.lisp", "FOLDER", folder)

	# # Compute with LISP IDyOM

	# os.system("sbcl --noinform --load lisp/compute.lisp")

	# replaceinFile("lisp/compute.lisp", folder, "FOLDER")


	#folder = "lisp/midis/"
	#folder = "dataset/bach_sub/"

	# Our IDyOM
	now = time.time()
	likelihoods1, files1 = cross_validation(folder, maxOrder=20, quantization=24, k_fold=10) #k-fold=10
	print("execution:", time.time()-now)

	# LISP version

	L2 = lisp.getDico("lisp/12-cpitch_onset-cpitch_onset-nil-nil-melody-nil-10-both-nil-t-nil-c-nil-t-t-x-3.dat")

	likelihoods2, files2 = lisp.getLikelihoods(L2)

	likelihood2 = np.mean(likelihoods2), np.std(likelihoods2), len(likelihoods2)

	plt.ylabel("Likelihood")
	plt.bar([0, 1], [np.mean(likelihoods1), likelihood2[0]], color="b", yerr=[1.96*np.std(likelihoods1)/np.sqrt(len(likelihoods1)), 1.96*likelihood2[1]/np.sqrt(likelihood2[2])])

	if not SERVER:
		plt.show()
	else:
		plt.savefig("figs/server/Lisp/likelihood.eps")
		plt.close()

	# Comparing models on pieces

	M = data.data()
	M.parse(folder, augment=False)
	dat1, files3 = M.getScoresFeatures()

	dico = dict(zip(files1, likelihoods1))

	dico2 = dict(zip(files2, likelihoods2))

	x1 = []
	x2 = []

	for file in files1:
		if file in dico2 and dico[file] is not None and dico2[file] is not None:
			x1.append(dico[file])
			x2.append(dico2[file])

	compareLikelihoods(x1, x2, name="Lisp/compareLikelihoods")



	# ploting in the music space

	dat2, files4 = M.getScoresFeatures()

	dico2 = dict(zip(files2, likelihoods2))

	weights = []
	colors = []

	for file in files1:
		if file in dico2 and dico2[file] is not None :
			weights.append(500*abs(dico[file]-dico2[file])**2)
			if dico[file]-dico2[file] < 0:
				colors.append('coral')
			elif dico[file]-dico2[file] > 0:
				colors.append('deepskyblue')
			else:
				colors.append('black')
		else:
			weights.append(10)
			colors.append('black')

	plt.scatter(dat2[0][:len(dat2[1])],dat2[1], s=weights, c=colors)

	plt.title('Python - Lisp')
	plt.xlabel('Average 1-note interval')
	plt.ylabel('Average note onset')

	if not SERVER:
		plt.show()
	else:
		plt.savefig("figs/server/Lisp/scoreSpace.eps")
		plt.close()

	# LATER 
	quit()
	plt.ylabel("Likelihood")
	plt.xlabel("time")
	plt.plot(L2['1']["probability"])
	plt.plot(L.getLikelihoodfromFile(folder+L2['1']["melody.name"][0][1:-1] + ".mid"))
	plt.show()


def Train(folder, jump=False):

	L = idyom.idyom(jump=jump, maxOrder=20)
	M = data.data(quantization=24)
	M.parse(folder)
	L.train(M)

	if folder[-1] == "/":
		folder = folder[:-1]

	L.save("models/"+str(folder[folder.rfind("/")+1:])+"_jump_"+str(jump)+".model")

def LikelihoodOverFolder(folderTrain, folder, jump=False, zero_padding=True):
	L = idyom.idyom(jump=jump)

	if folderTrain[-1] == "/":
		folderTrain = folderTrain[:-1]

	if os.path.isfile("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) +"_jump_"+str(jump)+".model"):
		print("We load saved model.")
		L.load("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) +"_jump_"+str(jump)+".model")
	else:
		print("No saved model found, please train before.")
		print("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) +"_jump_"+str(jump)+".model")
		quit()

	S, files = L.getSurprisefromFolder(folder)

	data = {}

	for i in range(len(S)):
		name = files[i][files[i].rfind("/")+1:files[i].rfind(".")]
		data[name] = np.array(S[i])

	if not os.path.exists(folder+"surprises"):
		os.makedirs(folder+"surprises")

	sio.savemat(folder+'surprises/surpriseSignal_'+str(folderTrain[folderTrain.rfind("/")+1:])+'_jump_'+str(jump)+'.mat', data)
	pickle.dump(data, open(folder+'surprises/surpriseSignal_'+str(folderTrain[folderTrain.rfind("/")+1:])+'_jump_'+str(jump)+'.pickle', "wb" ) )

	print()
	print()
	print()
	print("Data have been succesfully saved in:",folder+'surprises/surpriseSignal_'+str(folderTrain[folderTrain.rfind("/")+1:])+'_jump_'+str(jump)+'.mat')
	print("Including a .mat for matlab purpose and a .pickle for python purpose.")
	print()
	print()

	if not SERVER:
		for i in range(len(S)):
			plt.title(files[i])
			plt.plot(S[i])
			plt.show()
			#print(S[i])

def main():
	"""
	Call this method to easily use the program.
	"""

	pass

if __name__ == "__main__":

	usage = "usage %prog [options]"
	parser = OptionParser(usage)

	# parser.add_option("-t", "--test", type="int",
	# 				  help="1 if you want to launch unittests",
	# 				  dest="tests", default=0)

	# parser.add_option("-o", "--opti", type="string",
	# 				  help="launch optimisation of hyper parameters on the passed dataset",
	# 				  dest="hyper", default="")

	# parser.add_option("-c", "--check", type="string",
	# 				  help="check the passed dataset",
	# 				  dest="check", default="")

	# parser.add_option("-g", "--generate", type="int",
	# 				  help="generate piece of the passed length",
	# 				  dest="generate", default=0)

	# parser.add_option("-s", "--surprise", type="string",
	# 				  help="return the surprise over a given dataset",
	# 				  dest="surprise", default="")

	# parser.add_option("-l", "--lisp", type="string",
	# 				  help="plot comparison with the lisp version",
	# 				  dest="lisp", default="")

	# parser.add_option("-j", "--jump", type="string",
	# 				  help="plot comparison with the jump",
	# 				  dest="jump", default="")

	# parser.add_option("-p", "--plot", type="string",
	# 				  help="plot likelihood of idyom model",
	# 				  dest="plot", default="")

	# parser.add_option("-k", "--k_fold", type="int",
	# 			  help="set the value of k for cross validation",
	# 			  dest="k", default=None)

	parser.add_option("-a", "--ajump", type="string",
					  help="plot comparison with the jump",
					  dest="ajump", default="")

	parser.add_option("-t", "--train", type="string",
				  help="Train the model with the passed folder",
				  dest="train_folder", default=None)

	parser.add_option("-j", "--jump", type="int",
				  help="Use JUMP model as LTM is 1 is passed",
				  dest="jump", default=0)

	parser.add_option("-l", "--likelihood", type="string",
				  help="Compute likelihoods over the passed folder",
				  dest="trial_folder", default=None)

	parser.add_option("-z", "--zero_padding", type="int",
				  help="Specify if you want to use zero padding in the surprise output (1 by default)",
				  dest="zero_padding", default=1)

	parser.add_option("-p", "--lisp", type="string",
					  help="plot comparison with the lisp version",
					  dest="lisp", default="")

	parser.add_option("-i", "--in", type="string",
					  help="Training folder to use",
					  dest="folderTrain", default="bachMelodies")

	options, arguments = parser.parse_args()


	if options.train_folder is not None:
		Train(options.train_folder, jump=options.jump==1)

	if options.trial_folder is not None:
		folderTrain = options.folderTrain
		if options.train_folder is not None:
			folderTrain = options.train_folder
		LikelihoodOverFolder(folderTrain, options.trial_folder, jump=options.jump==1, zero_padding=options.zero_padding==1)

	if options.ajump != "":	
		compareJump(options.ajump)

	if options.lisp != "":	
		compareWithLISP(options.lisp)
	
	# if options.tests == 1:
	# 	loader = unittest.TestLoader()

	# 	start_dir = "unittests/"
	# 	suite = loader.discover(start_dir)

	# 	runner = unittest.TextTestRunner()
	# 	runner.run(suite)

	# if options.hyper != "":
	# 	L = idyom.idyom(maxOrder=30)

	# 	L.benchmarkQuantization(options.hyper,train=0.8)
	# 	L.benchmarkOrder(options.hyper, 24, train=0.8)

	# if options.check != "":
	# 	checkDataSet(options.check)

	# if options.generate != 0:		
	# 	L = idyom.idyom(maxOrder=30)
	# 	M = data.data(quantization=6)
	# 	#M.parse("../dataset/")
	# 	M.parse("dataset/")
	# 	L.train(M)
	# 	s = L.generate(int(options.generate))
	# 	s.plot()
	# 	s.writeToMidi("exGen.mid")

	# if options.jump != "":		
	# 	compareJump(options.jump)

	# if options.surprise != "":
	# 	L = idyom.idyom(maxOrder=30)
	# 	M = data.data(quantization=6)
	# 	#M.parse("../dataset/")
	# 	M.parse("dataset/")
	# 	L.train(M)

	# 	S = L.getSurprisefromFile(options.surprise, zero_padding=True)

	# 	plt.plot(S)
	# 	plt.xlabel("Time in quantization step")
	# 	plt.ylabel("Expected surprise (-log2(p))")
	# 	plt.show()

	# 	print(S)

	# if options.lisp != "":		
	# 	compareWithLISP(options.lisp)

	# if options.plot != "":
	# 	if options.k is None:		
	# 		plotLikelihood(options.plot)
	# 	else:
	# 		plotLikelihood(options.plot, k_fold=options.k)

