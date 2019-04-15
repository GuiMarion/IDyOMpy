import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def toFloat(f):
	try:
		return float(f)
	except ValueError:
		return f

def getDico(file):
	f=open(file, "r")
	lines = f.readlines()
	dico = {}
	keys = lines[0].split()
	for i in tqdm(range(1, len(lines))):
		tmp = lines[i].split()

		if tmp[1] not in dico:
			dico[tmp[1]] = {}

		k = 0
		for key in keys:
			if key not in dico[tmp[1]]:
				dico[tmp[1]][key] = []
			dico[tmp[1]][key].append(toFloat(tmp[k]))
			k += 1

	return dico

def getLikelihood(D):

	likelihood = []

	for melody_id in D:
		tmp = []
		for note in D[melody_id]["probability"]:
			tmp.append(note)

		likelihood.append(np.mean(tmp))

	return np.mean(likelihood), np.std(likelihood)
	print("likelihood:", np.mean(likelihood), "| std:", np.std(likelihood))

#D = getDico("2-cpitch_onset-cpitch_onset-nil-nil-melody-nil-10-both+-nil-t-nil-c-nil-t-t-x-3.dat")
#plt.plot(D['1']["probability"])
#plt.show()
