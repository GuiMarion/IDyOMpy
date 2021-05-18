import pickle
import sys 
import matplotlib.pyplot as plt

file1 = str(sys.argv[1])
file2 = str(sys.argv[2])


aa = pickle.load(open(file1, "rb"))
bb = pickle.load(open(file2, "rb"))


if aa.keys() != bb.keys():
	print("The two files does not contain the same pieces...")
	exit()

differences = 0
for key in aa.keys():
	if aa[key] != bb[key]:
		print(key +" is different for the two models.")
		plt.plot(aa[key], label="First Model")
		plt.plot(bb[key], label="Second Model")
		plt.title(key)
		plt.legend()
		plt.show()
		differences += 1

if differences == 0:
	print("The two models are identical!")
