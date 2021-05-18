import matplotlib.pyplot as plt
import pickle
import sys

file = str(sys.argv[1])

aa = pickle.load(open(file, "rb"))
for key in aa:
	plt.plot(aa[key], label=key)
plt.legend()
plt.show()



