"""
Enter point of the program.
"""
from idyom import idyom
from idyom import data
from lisp import parser as lisp

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

SERVER = False

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
	Does not delete automatically!
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

def cross_validation(folder, k_fold=10, maxOrder=20, quantization=24, time_representation=False, \
										zero_padding=True, long_term_only=False, short_term_only=False,\
										viewPoints="both"):
	"""

	"""
	if viewPoints == "pitch":
		viewPoints_o = ["pitch"]
	elif viewPoints == "length":
		viewPoints_o = ["length"]
	elif viewPoints == "both":
		viewPoints_o = ["pitch", "length"]


	np.random.seed(0)

	Likelihoods = []

	files = []
	for filename in glob(folder + '/**', recursive=True):
		if filename[filename.rfind("."):] in [".mid", ".midi"]:
			files.append(filename)

	np.random.shuffle(files)

	if int(k_fold) == -1:
		k_fold = len(files)

	if int(k_fold) > len(files):
		raise ValueError("Cannot process with k_fold greater than number of files. Please use -k options to specify a smaller k for cross validation.")

	k_fold = len(files) // int(k_fold)

	validationFiles = []

	for i in tqdm(range(math.ceil(len(files)/k_fold))):
		trainData = files[:i*k_fold] + files[(i+1)*k_fold:]
		evalData = files[i*k_fold:(i+1)*k_fold]

		# Our IDyOM
		L = idyom.idyom(maxOrder=maxOrder, viewPoints=viewPoints_o)
		M = data.data(quantization=quantization)
		M.addFiles(trainData)

		L.train(M)

		for file in evalData:
			tmp = L.getLikelihoodfromFile(file, long_term_only=long_term_only, short_term_only=short_term_only)
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

	if not os.path.exists("lisp/midis/"):
		os.makedirs("lisp/midis/")

	os.system("rm -rf lisp/midis/*")

	# Add folder to lisp database

	replaceinFile("lisp/compute.lisp", "FOLDER", folder)

	# Compute with LISP IDyOM

	os.system("sbcl --noinform --load lisp/compute.lisp")

	replaceinFile("lisp/compute.lisp", folder, "FOLDER")


	folder = "lisp/midis/"
	folder = "dataset/bach_sub/"

	# Our IDyOM
	now = time.time()
	likelihoods1, files1 = cross_validation(folder, maxOrder=20, quantization=24, k_fold=5) #k-fold=10
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

	print("IDyOMpy:",likelihoods1)
	print("LISP:",likelihoods2)

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


def Train(folder, k_fold=5, quantization=24, maxOrder=20, time_representation=False, \
				zero_padding=True, long_term_only=False, short_term_only=False, viewPoints="both"):

	if folder[-1] == "/":
		folder = folder[:-1]

	if viewPoints == "pitch":
		viewPoints_o = ["pitch"]
	elif viewPoints == "length":
		viewPoints_o = ["length"]
	elif viewPoints == "both":
		viewPoints_o = ["pitch", "length"]
	else:
		raise("We do not know these viewpoints ... ")

	if os.path.isfile("models/"+ str(folder[folder.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder) +"_viewpoints_"+str(viewPoints)+ ".model"):
		print("There is already a model saved for these data, would you like to train again? (y/N)\n")
		rep = input("")
		while rep not in ["y", "Y", "n", "N", "", "\n"]:
			rep = input("We did not understand, please type again (y/N).")

		if rep.lower() == "y":
			pass
		else:
			return

	start = time.time()
	L = idyom.idyom(maxOrder=maxOrder, viewPoints=viewPoints_o)
	M = data.data(quantization=quantization)
	M.parse(folder)
	print("Training Started, data processing: "+str(time.time()-start))
	start = time.time()
	L.train(M)
	print("Training Ended, training: " + str(time.time()-start))

	L.save("models/"+ str(folder[folder.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+str(viewPoints)+ ".model")

def SurpriseOverFolder(folderTrain, folder, k_fold=5, quantization=24, maxOrder=20, time_representation=False, \
											zero_padding=True, long_term_only=False, short_term_only=False,\
											viewPoints="both"):
	
	L = idyom.idyom()

	if folderTrain[-1] == "/":
		folderTrain = folderTrain[:-1]

	if folder[-1] != "/":
		folder += "/"

	name_train = folderTrain[folderTrain[:-1].rfind("/")+1:] + "/"

	name = folder[folder[:-1].rfind("/")+1:]

	if not os.path.exists("out/"+name):
	    os.makedirs("out/"+name)

	if not os.path.exists("out/"+name+"surprises/"):
	    os.makedirs("out/"+name+"surprises/")

	if not os.path.exists("out/"+name+"surprises/"+name_train):
	    os.makedirs("out/"+name+"surprises/"+name_train)

	if not os.path.exists("out/"+name+"surprises/"+name_train+"data/"):
	    os.makedirs("out/"+name+"surprises/"+name_train+"data/")

	if not os.path.exists("out/"+name+"surprises/"+name_train+"figs/"):
	    os.makedirs("out/"+name+"surprises/"+name_train+"figs/")


	if os.path.isfile("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+str(viewPoints) + ".model"):
		print("We load saved model.")
		L.load("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+str(viewPoints) + ".model")
	else:
		print("No saved model found, please train before.")
		print("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+str(viewPoints) + ".model")
		quit()

	S, files = L.getSurprisefromFolder(folder, time_representation=time_representation, long_term_only=long_term_only, short_term_only=short_term_only)

	data = {}

	for i in range(len(S)):
		name_tmp = files[i][files[i].rfind("/")+1:files[i].rfind(".")]
		name_tmp = name_tmp.replace("-", "_")
		data[name_tmp] = np.array(S[i]).tolist()

	more_info = ""
	if long_term_only:
		more_info += "_longTermOnly"
	if short_term_only:
		more_info += "_shortTermOnly" 

	more_info += "_quantization_"+str(quantization) + "_maxOrder_"+str(maxOrder)+"_viewpoints_"+str(viewPoints)


	sio.savemat("out/"+name+"surprises/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.mat', data)
	pickle.dump(data, open("out/"+name+"surprises/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.pickle', "wb" ) )

	print()
	print()
	print()
	print("Data have been succesfully saved in:","out/"+name+"surprises/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.mat')
	print("Including a .mat for matlab purpose and a .pickle for python purpose.")
	print()
	print()

	if not os.path.exists("out/"+name+"surprises/"+name_train+"figs/"+more_info[1:]):
	    os.makedirs("out/"+name+"surprises/"+name_train+"figs/"+more_info[1:])

	for i in range(len(S)):
		plt.title(files[i])
		plt.plot(S[i])
		plt.savefig("out/"+name+"surprises/"+name_train+"figs/"+more_info[1:]+"/"+str(files[i][files[i].rfind("/")+1:files[i].rfind(".")])+'.eps')
		if not SERVER:
			plt.show()
		else:
			plt.close()

def SilentNotesOverFolder(folderTrain, folder, threshold=0.3, k_fold=5, quantization=24, maxOrder=20, time_representation=False, \
											zero_padding=True, long_term_only=False, short_term_only=False, viewPoints="both"):
	
	L = idyom.idyom()

	if folderTrain[-1] == "/":
		folderTrain = folderTrain[:-1]

	if folder[-1] != "/":
		folder += "/"

	name_train = folderTrain[folderTrain[:-1].rfind("/")+1:] + "/"

	name = folder[folder[:-1].rfind("/")+1:]

	if not os.path.exists("out/"+name):
	    os.makedirs("out/"+name)

	if not os.path.exists("out/"+name+"missing_notes/"):
	    os.makedirs("out/"+name+"missing_notes/")

	if not os.path.exists("out/"+name+"missing_notes/"+name_train):
	    os.makedirs("out/"+name+"missing_notes/"+name_train)

	if not os.path.exists("out/"+name+"missing_notes/"+name_train+"data/"):
	    os.makedirs("out/"+name+"missing_notes/"+name_train+"data/")

	if not os.path.exists("out/"+name+"missing_notes/"+name_train+"figs/"):
	    os.makedirs("out/"+name+"missing_notes/"+name_train+"figs/")


	if os.path.isfile("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder) +"_viewpoints_"+str(viewPoints)+ ".model"):
		print("We load saved model.")
		L.load("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+str(viewPoints) + ".model")
	else:
		print("No saved model found, please train before.")
		print("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+str(viewPoints) + ".model")
		quit()

	S, files = L.getDistributionsfromFolder(folder, threshold, time_representation=time_representation, long_term_only=long_term_only, short_term_only=short_term_only)

	data = {}

	for i in range(len(S)):
		name_tmp = files[i][files[i].rfind("/")+1:files[i].rfind(".")]
		name_tmp = name_tmp.replace("-", "_")
		data[name_tmp] = np.array(S[i]).tolist()

	more_info = ""
	if long_term_only:
		more_info += "_longTermOnly"
	if short_term_only:
		more_info += "_shortTermOnly" 

	more_info += "_quantization_"+str(quantization) + "_maxOrder_"+str(maxOrder)+"_viewpoints_"+str(viewPoints)


	sio.savemat("out/"+name+"missing_notes/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.mat', data)
	pickle.dump(data, open("out/"+name+"missing_notes/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.pickle', "wb" ) )

	print()
	print()
	print()
	print("Data have been succesfully saved in:","out/"+name+"missing_notes/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.mat')
	print("Including a .mat for matlab purpose and a .pickle for python purpose.")
	print()
	print()

	if not os.path.exists("out/"+name+"missing_notes/"+name_train+"figs/"+more_info[1:]):
	    os.makedirs("out/"+name+"missing_notes/"+name_train+"figs/"+more_info[1:])

	for i in range(len(files)):
		plt.plot(S[i][0])
		plt.plot(S[i][1])
		plt.legend(["Actual Notes", "Missing Notes"])
		plt.title("Piece: " + files[i])
		plt.savefig("out/"+name+"missing_notes/"+name_train+"figs/"+more_info[1:]+"/"+str(files[i][files[i].rfind("/")+1:files[i].rfind(".")])+'.eps')
		if not SERVER:
			plt.show()
		else:
			plt.close()




def evaluation(folder, k_fold=5, quantization=24, maxOrder=20, time_representation=False, \
				zero_padding=True, long_term_only=False, short_term_only=False, viewPoints="both"):

	if folder[-1] != "/":
		folder += "/"

	name = folder[folder[:-1].rfind("/")+1:]


	if not os.path.exists("out/"+name):
	    os.makedirs("out/"+name)

	if not os.path.exists("out/"+name+"eval/"):
	    os.makedirs("out/"+name + "eval/")

	if not os.path.exists("out/"+name+"eval/data/"):
	    os.makedirs("out/"+name+"eval/data/")

	if not os.path.exists("out/"+name+"eval/figs/"):
	    os.makedirs("out/"+name+"eval/figs/")


	more_info = "_"
	if long_term_only:
		more_info += "long_term_only_"
	if short_term_only:
		more_info += "short_term_only_"

	more_info += "k_fold_"+str(k_fold)+"_quantization_"+str(quantization) + "_maxOrder_"+str(maxOrder)+"_viewpoints_"+str(viewPoints)

	likelihoods, files = cross_validation(folder, maxOrder=maxOrder, quantization=quantization, k_fold=k_fold, \
												long_term_only=long_term_only, short_term_only=short_term_only)

	plt.ylabel("Likelihood")
	plt.bar([0], [np.mean(likelihoods)], color="b", yerr=[1.96*np.std(likelihoods)/np.sqrt(len(likelihoods))])

	plt.savefig("out/"+name+'eval/figs/likelihoods_cross-eval'+more_info+'.eps')

	if not SERVER:
		plt.show()
	else:
		plt.close()

	print("IDyOMpy:",likelihoods)
	print("Mean:", np.mean(likelihoods))
	print("Std:", np.std(likelihoods))

	pickle.dump((likelihoods, files), open("out/"+name+'eval/data/likelihoods_cross-eval'+more_info+'.pickle', "wb" ) )

	print()
	print()
	print()
	print("Data (likelihoods) have been succesfully saved in:","out/"+name+'eval/data/likelihoods_cross-eval'+more_info+'.pickle')
	print()
	print("Figure have been succesfully saved in:","out/"+name+'eval/figs/likelihoods_cross-eval'+more_info+'.eps')
	print()

def main():
	"""
	Call this method to easily use the program.
	"""

	pass

if __name__ == "__main__":

	usage = "usage %prog [options]"
	parser = OptionParser(usage)

	# Create directory tree
	if not os.path.exists("out/"):
	    os.makedirs("out/")


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


	# parser.add_option("-p", "--plot", type="string",
	# 				  help="plot likelihood of idyom model",
	# 				  dest="plot", default="")

	# parser.add_option("-k", "--k_fold", type="int",
	# 			  help="set the value of k for cross validation",
	# 			  dest="k", default=None)


	parser.add_option("-a", "--test", type="int",
					  help="1 if you want to launch unittests",
					  dest="tests", default=0)

	parser.add_option("-t", "--train", type="string",
				  help="Train the model with the passed folder",
				  dest="train_folder", default=None)

	parser.add_option("-s", "--surprise", type="string",
				  help="Compute surprise over the passed folder. We use -t argument to train, if none are privided, we use the passed folder to cross-train.",
				  dest="trial_folder", default=None)

	parser.add_option("-n", "--silentNotes", type="string",
				  help="Compute silent notes probabilities over the passed folder. We use -t argument to train, if none are provided, we use the passed folder to cross-train.",
				  dest="trial_folder_silent", default=None)

	parser.add_option("-d", "--threshold_missing_notes", type="float",
				  help="Define the threshold for choosing the missing notes (0.3 by default)",
				  dest="threshold_missing_notes", default=0.3)

	parser.add_option("-z", "--zero_padding", type="int",
				  help="Specify if you want to use zero padding in the surprise output, enable time representation (default 0)",
				  dest="zero_padding", default=None)

	parser.add_option("-p", "--lisp", type="string",
					  help="plot comparison with the lisp version",
					  dest="lisp", default="")

	parser.add_option("-b", "--short_term", type="int",
					  help="Only use short term model (default 0)",
					  dest="short_term_only", default=0)

	parser.add_option("-c", "--cross_eval", type="string",
					  help="Compute likelihoods by pieces over the passed dataset using k-fold cross-eval.",
					  dest="cross_eval", default=None)

	parser.add_option("-l", "--long_term", type="int",
					  help="Only use long term model (default 0)",
					  dest="long_term_only", default=0)

	parser.add_option("-k", "--k_fold", type="int",
					  help="Specify the k-fold for all cross-eval, you can use -1 for leave-one-out (default 5).",
					  dest="k_fold", default=5)

	parser.add_option("-q", "--quantization", type="int",
					  help="Rythmic quantization to use (default 24).",
					  dest="quantization", default=24)

	parser.add_option("-v", "--viewPoints", type="string",
					  help="Viewpoints to use (pitch, length or both), default both",
					  dest="viewPoints", default="both")

	parser.add_option("-m", "--max_order", type="int",
					  help="Maximal order to use (default 20).",
					  dest="max_order", default=24)		

	parser.add_option("-r", "--check_dataset", type="string",
					  help="Check wether the passed folder contains duplicates.",
					  dest="folder_duplicates", default="")	

	options, arguments = parser.parse_args()

	if options.zero_padding is not None:
		time_representation = True
	else:
		time_representation = False

	if options.train_folder is not None:
		print("Training ...")
		Train(options.train_folder, k_fold=options.k_fold, quantization=options.quantization, maxOrder=options.max_order, \
									time_representation=time_representation, zero_padding=options.zero_padding==1, \
									long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
									viewPoints=options.viewPoints)

	if options.cross_eval is not None:
		print("Evaluation on", str(options.cross_eval), "...")
		evaluation(str(options.cross_eval), k_fold=options.k_fold, quantization=options.quantization, maxOrder=options.max_order, \
											time_representation=time_representation, zero_padding=options.zero_padding==1, \
											long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
											viewPoints=options.viewPoints)

	if options.trial_folder is not None:
		if options.train_folder is None:
			print("You did not provide a train folder, therefore, we will cross evaluation on the trial folder to train.")
			raise NotImplemented("This function is not implemented yet ...")

		SurpriseOverFolder(options.train_folder, options.trial_folder, \
							k_fold=options.k_fold,quantization=options.quantization, maxOrder=options.max_order, \
							time_representation=time_representation, zero_padding=options.zero_padding==1, \
							long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
							viewPoints=options.viewPoints)

	if options.trial_folder_silent is not None:
		if options.train_folder is None:
			print("You did not provide a train folder, therefore, we will cross evaluation on the trial folder to train.")
			raise NotImplemented("This function is not implemented yet ...")


		SilentNotesOverFolder(options.train_folder, options.trial_folder_silent, threshold=options.threshold_missing_notes, \
							k_fold=options.k_fold,quantization=options.quantization, maxOrder=options.max_order, \
							time_representation=time_representation, zero_padding=options.zero_padding==1, \
							long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
							viewPoints=options.viewPoints)
	if options.folder_duplicates != "":	
		checkDataSet(options.folder_duplicates)

	if options.lisp != "":	
		compareWithLISP(options.lisp)
	
	if options.tests == 1:
		loader = unittest.TestLoader()

		start_dir = "unittests/"
		suite = loader.discover(start_dir)

		runner = unittest.TextTestRunner()
		runner.run(suite)
