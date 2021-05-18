from glob import glob
import numpy as np
from shutil import copyfile
import os


def main(folder, trainp=0.8, dest="dataset/"):

	files = []

	for file in glob(folder+"*"):
		if file[file.rfind("."):] == ".krn":
			files.append(file)

	np.random.shuffle(files)

	train = files[:int(trainp*len(files))]
	test = files[int(trainp*len(files)):]

	if not os.path.exists(dest):
	    os.makedirs(dest)

	if not os.path.exists(dest+"train/"):
	    os.makedirs(dest+"train/")

	if not os.path.exists(dest+"test/"):
	    os.makedirs(dest+"test/")

	for file in train:
		copyfile(file, dest+"train/"+file[file.rfind("/"):])

	for file in test:
		copyfile(file, dest+"test/"+file[file.rfind("/"):])


main("chorales_krn/")
