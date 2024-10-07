"""
Enter point of the program.
"""
from idyom import idyom
from idyom import data

from optparse import OptionParser
from shutil import copyfile
from shutil import rmtree
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
import random
import ast

SERVER = False

if SERVER:
	plt.ioff()

def foo_callback(option, opt, value, parser):
	setattr(parser.values, option.dest, value.split(','))

def vstr(L):
	ret = ""
	for elem in L:
		ret += str(elem) + "_"


	return ret[:-1]

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
										viewPoints=["pitch", "length"], genuine_entropies=False, use_original_PPM=False):
	"""
	Cross-validate by training on on k-1 folds of the folder and evaluate on the remaining fold
	k_fold = -1 means leave-one-out 
	"""
	np.random.seed(0)

	viewPoints_o = []
	for elem in viewPoints:
		if elem in data.AVAILABLE_VIEWPOINTS:
			viewPoints_o.append(elem)
		else:
			print("I don't know the viewpoint: " + str(elem))
			print("Please check and rerun the program.")
			quit()

	ICs = []
	Entropies = []

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
	Likelihoods = []

	for i in tqdm(range(math.ceil(len(files)/k_fold))):
		trainData = files[:i*k_fold] + files[(i+1)*k_fold:]
		evalData = files[i*k_fold:(i+1)*k_fold]

		L = idyom.idyom(maxOrder=maxOrder, viewPoints=viewPoints_o, use_original_PPM=use_original_PPM)
		M = data.data(quantization=quantization)
		M.addFiles(trainData)

		L.train(M)

		for file in evalData:
			IC, E = L.getSurprisefromFile(file, long_term_only=long_term_only, short_term_only=short_term_only, time_representation=time_representation, zero_padding=zero_padding, genuine_entropies=genuine_entropies)
			ICs.append(IC)
			Entropies.append(E)
			filename = file[file.rfind("/")+1:file.rfind(".")]
			filename = filename.replace("-", "_")
			validationFiles.append(filename)

	return ICs, Entropies, validationFiles


def Train(folder, quantization=24, maxOrder=20, time_representation=False, \
				zero_padding=True, long_term_only=False, short_term_only=False, \
				viewPoints=["pitch, length"], use_original_PPM=False):

	'''
	Train a model with the midi files contained in the passed folder.
	'''

	if folder[-1] == "/":
		folder = folder[:-1]

	viewPoints_o = []
	for elem in viewPoints:
		if elem in data.AVAILABLE_VIEWPOINTS:
			viewPoints_o.append(elem)
		else:
			print("I don't know the viewpoint: " + str(elem))
			print("Please check and rerun the program.")
			quit()

	ppm_name = "_originalPPM" if use_original_PPM else ""
	if os.path.isfile("models/"+ str(folder[folder.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder) +"_viewpoints_"+vstr(viewPoints)+ str(ppm_name) + ".model"):
		print("There is already a model saved for these data, would you like to train again? (Y/N)\n")
		rep = input("")
		while rep not in ["y", "Y", "n", "N", "", "\n"]:
			rep = input("We did not understand, please type again (Y/N).")

		if rep.lower() == "y":
			pass
		else:
			return

	preComputeEntropies = not (long_term_only or short_term_only) # We only precompute if we need to combine short and long term models

	L = idyom.idyom(maxOrder=maxOrder, viewPoints=viewPoints_o, use_original_PPM=use_original_PPM)
	M = data.data(quantization=quantization)
	M.parse(folder, augment=True)
	L.train(M)

	ppm_name = "_originalPPM" if use_original_PPM else ""
	L.save("models/"+ str(folder[folder.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints)+ppm_name+ ".model")


def Train_by_piece(folder, nb_pieces=20, quantization=24, maxOrder=20, time_representation=False, \
				zero_padding=True, long_term_only=False, short_term_only=False, viewPoints=["pitch", "length"], \
				intialization="", use_original_PPM=False):

	'''
	Train iteratively on the passed folder (can be also initialized with western data)
	and generate the prediction error for each piece along the training. This allows
	to look at the dynamics of learning during training. 
	'''

	name_temp_file = ".tmp_test_folder_" + folder[folder.rfind("/")+1:] + "_" + str(np.random.randint(100, 999))

	if folder[-1] == "/":
		folder = folder[:-1]

	viewPoints_o = []
	for elem in viewPoints:
		if elem in data.AVAILABLE_VIEWPOINTS:
			viewPoints_o.append(elem)
		else:
			print("I don't know the viewpoint: " + str(elem))
			print("Please check and rerun the program.")
			quit()

	L = idyom.idyom(maxOrder=maxOrder, viewPoints=viewPoints_o, evolutive=True, use_original_PPM=use_original_PPM)

	files = glob(folder+'/**.mid', recursive=True) + glob(folder+'/**.midi', recursive=True)

	random.shuffle(files)
	train = files[:-nb_pieces]
	test = files[-nb_pieces:]

	if intialization != "":
		europe_files = files = glob(intialization, recursive=True)
		train = europe_files[:200] + train

	if os.path.exists(name_temp_file):
		if os.path.isdir(name_temp_file):
			rmtree(name_temp_file)
		else:
			os.remove(name_temp_file)
	os.mkdir(name_temp_file)
	
	for file in test: 
		copyfile(file, name_temp_file+file[file.rfind("/"):])

	try:
		note_counter = []
		dicos = []
		matrix = np.zeros((len(train), nb_pieces))
		print("___ Starting Training ___")
		k = 0
		for file in tqdm(train):
			try:
				M = data.data(quantization=quantization)
				M.parseFile(file)
				L.train(M, preComputeEntropies=False)

				S, E, files = L.getSurprisefromFolder(name_temp_file, time_representation=time_representation, long_term_only=long_term_only, short_term_only=short_term_only)
				note_counter.append(len(M.viewPointRepresentation["pitch"][0]))

				dico = {}
				for i in range(len(files)):
					dico[files[i][files[i].rfind("/")+1:]] = S[i]

				dicos.append(dico)
				tmp = []
				for s in S:
					tmp.append(np.mean(s))

				matrix[k,:] = tmp
				k += 1
			except (FileNotFoundError, RuntimeError, ValueError):
				print(file+ " skipped.")

		for i in range(1, len(note_counter)):
			note_counter[i] += note_counter[i-1] 

		saving = {}
		saving['matrix'] = matrix
		saving['note_counter'] = note_counter
		saving['dico'] = dico

		if not os.path.exists("out/"+folder[folder.rfind("/"):]):
		    os.makedirs("out/"+folder[folder.rfind("/"):])

		if not os.path.exists("out/"+folder[folder.rfind("/"):]+"/evolution/"):
		    os.makedirs("out/"+folder[folder.rfind("/"):]+"/evolution/")

		pickle.dump(saving, open("out/"+folder[folder.rfind("/")+1:]+"/evolution/"+folder[folder.rfind("/")+1:]+'.pickle', "wb" ) )
		print("Data saved at " +"out/"+folder[folder.rfind("/")+1:]+"/evolution/"+folder[folder.rfind("/")+1:]+'.pickle')
		sio.savemat("out/"+folder[folder.rfind("/")+1:]+"/evolution/"+folder[folder.rfind("/")+1:]+'.mat', saving)
		print("And at " +"out/"+folder[folder.rfind("/")+1:]+"/evolution/"+folder[folder.rfind("/")+1:]+'.mat')
		rmtree(name_temp_file)
	except Exception as e:
		rmtree(name_temp_file)


def SurpriseOverFolder(folderTrain, folder, k_fold=5, quantization=24, maxOrder=20, time_representation=False, \
											zero_padding=True, long_term_only=False, short_term_only=False,\
											viewPoints=["pitch", "length"], genuine_entropies=False, use_original_PPM=False):
	
	'''
	Train a model (or load it if already saved) and evaluate it on the passed folder.
	Computed the surprise signal for each file in the folder
	'''

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

	ppm_name = "_originalPPM" if use_original_PPM else ""
	if os.path.isfile("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ppm_name + ".model"):
		print("We load saved model.")
		L.load("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ppm_name + ".model")
	else:
		print("No saved model found, please train before.")
		print("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ppm_name + ".model")
		quit()

	S, E, files = L.getSurprisefromFolder(folder, time_representation=time_representation, long_term_only=long_term_only, short_term_only=short_term_only, genuine_entropies=genuine_entropies)

	data = {}

	for i in range(len(S)):
		name_tmp = files[i][files[i].rfind("/")+1:files[i].rfind(".")]
		name_tmp = name_tmp.replace("-", "_")
		data[name_tmp] = [np.array(S[i]).tolist(), np.array(E[i]).tolist()]
	data["info"] = "Each variable corresponds to a song. For each song you have the Information Content as the first dimension, and then the Relative Entropy as the second dimension. They are both vectors over the time dimension."

	more_info = ""
	if long_term_only:
		more_info += "_longTermOnly"
	if short_term_only:
		more_info += "_shortTermOnly" 
	
	ppm_name = "_originalPPM" if use_original_PPM else ""
	more_info += "_quantization_"+str(quantization) + "_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ppm_name


	sio.savemat("out/"+name+"surprises/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.mat', data)
	pickle.dump(data, open("out/"+name+"surprises/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.pickle', "wb" ) )

	print()
	print()
	print()
	print("Data have been succesfully saved in:","out/"+name+"surprises/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.mat')
	print("Including a .mat for matlab purpose and a .pickle for python purpose.")
	print()
	print()

def SilentNotesOverFolder(folderTrain, folder, threshold=0.3, k_fold=5, quantization=24, maxOrder=20, time_representation=False, \
											zero_padding=True, long_term_only=False, short_term_only=False, \
											viewPoints=["pitch", "length"], use_original_PPM=False):
	
	'''
	Function used in The music of silence. Part II: Cortical Predictions during Silent Musical Intervals (https://www.jneurosci.org/content/41/35/7449)
	It computes the probabily to have a note in each natural musical silences (using only the duration/rythm dimension). 
	'''

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


	if os.path.isfile("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder) +"_viewpoints_"+vstr(viewPoints)+ ".model"):
		print("We load saved model.")
		L.load("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ".model")
	else:
		print("No saved model found, please train before.")
		print("models/"+ str(folderTrain[folderTrain.rfind("/")+1:]) + "_quantization_"+str(quantization)+"_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ".model")
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

	ppm_name = "_originalPPM" if use_original_PPM else ""
	more_info += "_quantization_"+str(quantization) + "_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ppm_name


	sio.savemat("out/"+name+"missing_notes/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.mat', data)
	pickle.dump(data, open("out/"+name+"missing_notes/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.pickle', "wb" ) )

	print()
	print()
	print()
	print("Data (surprises/IC)have been succesfully saved in:","out/"+name+"missing_notes/"+name_train+"data/"+str(folderTrain[folderTrain.rfind("/")+1:])+more_info+'.mat')
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
				zero_padding=True, long_term_only=False, short_term_only=False, viewPoints="both", genuine_entropies=False, use_original_PPM=False):

	'''
	Main function for the cross-validation
	'''
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

	ppm_name = "_originalPPM" if use_original_PPM else ""
	more_info += "k_fold_"+str(k_fold)+"_quantization_"+str(quantization) + "_maxOrder_"+str(maxOrder)+"_viewpoints_"+vstr(viewPoints) + ppm_name

	S, E, files = cross_validation(folder, maxOrder=maxOrder, quantization=quantization, k_fold=k_fold, time_representation=time_representation, \
												long_term_only=long_term_only, short_term_only=short_term_only, genuine_entropies=genuine_entropies, use_original_PPM=use_original_PPM)
	data = {}
	for i in range(len(S)):
		data[files[i]] = np.array(S[i]).tolist()
		data[files[i]] = [np.array(S[i]).tolist(), np.array(E[i]).tolist()]
	data["info"] = "Each variable corresponds to a song. For each song you have the Information Content as the first dimension, and then the Relative Entropy as the second dimension. They are both vectors over the time dimension."



	sio.savemat("out/"+name+'eval/data/likelihoods_cross-eval'+more_info+'.mat', data)
	pickle.dump(data, open("out/"+name+'eval/data/likelihoods_cross-eval'+more_info+'.pickle', "wb" ) )

	print()
	print()
	print()
	print("Data have been succesfully saved in:","out/"+name+'eval/data/likelihoods_cross-eval'+more_info+'.pickle')
	print("Including a .mat for matlab purpose and a .pickle for python purpose.")
	print()
	print()


if __name__ == "__main__":

	usage = "usage %prog [options]"
	parser = OptionParser(usage)

	# Create directory tree
	if not os.path.exists("out/"):
	    os.makedirs("out/")

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
				  help="Define the threshold for choosing the missing notes (0.2 by default)",
				  dest="threshold_missing_notes", default=0.2)

	parser.add_option("-z", "--zero_padding", type="int",
				  help="Specify if you want to use zero padding in the surprise output, enable time representation (default 0)",
				  dest="zero_padding", default=None)

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

	parser.add_option("-v", "--viewPoints", type= "string", action='callback', callback=foo_callback,
					  help="Viewpoints to use: pitch, length, interval and velocity, separate them with comas, default pitch,length.",
					  dest="viewPoints", default=['pitch', 'length'])

	parser.add_option("-m", "--max_order", type="int",
					  help="Maximal order to use (default 20).",
					  dest="max_order", default=20)		

	parser.add_option("-g", "--genuine_entropies", type="int",
					  help="Use this parameter to NOT use the entropy approximation. It takes longer (5 times) to compute but generate the genuine entropies, not an approximation (default 0).",
					  dest="genuine_entropies", default=0)		

	parser.add_option("-r", "--check_dataset", type="string",
					  help="Check wether the passed folder contains duplicates.",
					  dest="folder_duplicates", default="")	

	parser.add_option("-e", "--evolution", type="string",
					  help="Train and evaluate over training on the passed folder (cross-val).",
					  dest="train_test_folder", default=None)	

	parser.add_option("-i", "--init_evolution", type="string",
					  help="Folder to initialize the evolution on.",
					  dest="intialization", default="")	

	parser.add_option("-p", "--nb_pieces", type="int",
					  help="Number of pieces to evaluate on during evolution training.",
					  dest="nb_pieces", default=20)

	parser.add_option("-o", "--original_PPM", type="int",
					  help="Use the original Prediction by Partial Matching (PPM) mothod C",
					  dest="use_original_PPM", default=0)


	options, arguments = parser.parse_args()
	options.lisp = "" # Temporary

	if options.zero_padding is not None:
		time_representation = True
	else:
		time_representation = False

	if options.train_test_folder is not None:
		print("Evolution Training ...")
		Train_by_piece(options.train_test_folder, nb_pieces=options.nb_pieces, quantization=options.quantization, maxOrder=options.max_order, \
									time_representation=time_representation, zero_padding=options.zero_padding==1, \
									long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
									viewPoints=options.viewPoints, intialization=options.intialization, use_original_PPM=options.use_original_PPM==1)		

	if options.train_folder is not None:

		isValid = type(options.viewPoints) is type(["", ""])
		if isValid:
			for elem in options.viewPoints:
				if type(elem) is not type(""):
					isValid = False

		if not isValid: 
			print()
			print("The viewpoints you gave: " + str(options.viewPoints)+". Are not in the correct format. Please separate them with comas without spaces. Example: length,pitch.")
			quit()

		if options.viewPoints[-1] == "":
			print("Please do not use spaces between the viewpoint parameters.")
			quit()

		print("Viewpoints to use: "+ str(options.viewPoints))

		print("Training ...")
		Train(options.train_folder, quantization=options.quantization, maxOrder=options.max_order, \
									time_representation=time_representation, zero_padding=options.zero_padding==1, \
									long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
									viewPoints=options.viewPoints, use_original_PPM=options.use_original_PPM==1)

	if options.cross_eval is not None:
		print("Evaluation on", str(options.cross_eval), "...")
		evaluation(str(options.cross_eval), k_fold=options.k_fold, quantization=options.quantization, maxOrder=options.max_order, \
											time_representation=time_representation, zero_padding=options.zero_padding==1, \
											long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
											viewPoints=options.viewPoints, genuine_entropies=options.genuine_entropies==1, use_original_PPM=options.use_original_PPM==1)

	if options.trial_folder is not None:
		if options.train_folder is None:
			print("You did not provide a train folder, please provide one or use the --cross_eval option to cross-evaluate this folder.")
			quit()
		SurpriseOverFolder(options.train_folder, options.trial_folder, \
							k_fold=options.k_fold,quantization=options.quantization, maxOrder=options.max_order, \
							time_representation=time_representation, zero_padding=options.zero_padding==1, \
							long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
							viewPoints=options.viewPoints, genuine_entropies=options.genuine_entropies==1, use_original_PPM=options.use_original_PPM==1)

	if options.trial_folder_silent is not None:
		if options.train_folder is None:
			print("You did not provide a train folder, please provide one or use the --cross_eval option to cross-evaluate this folder.")
			quit()


		SilentNotesOverFolder(options.train_folder, options.trial_folder_silent, threshold=options.threshold_missing_notes, \
							k_fold=options.k_fold,quantization=options.quantization, maxOrder=options.max_order, \
							time_representation=time_representation, zero_padding=options.zero_padding==1, \
							long_term_only=options.long_term_only==1, short_term_only=options.short_term_only==1,\
							viewPoints=options.viewPoints, use_original_PPM=options.use_original_PPM==1)

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
