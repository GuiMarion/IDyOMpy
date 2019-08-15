import sys 
sys.path.append('../')

from idyom import data
import numpy as np

import matplotlib.pyplot as plt

M = data.data()
M.addFile("../stimuli/Chinese/test1/chinese-001.mid")

print(list(M.getData("length")[0]))
print(len(M.getData("length")[0]))

quit()

for i in range(1, 11):
	M = data.data()
	M.addFile("../stimuli/giovanni/audio"+str(i)+".mid")
	#M.addFile("../../My_First_Score.mid")
	#M.addFile("../dataset/bach_Pearce/chor-060.mid")
	#M.addFile("../dataset/bachMelodies/519_bwv1078.mid")
	#M.addFile("../dataset/bachMelodies/109_fugue4.mid")

	print(list(M.getData("length")[0]))

	freq = {}

	for elem in M.getData("length")[0]:
		if elem not in freq:
			freq[elem] = 0
		freq[elem] += 1

	print(freq)
	for key in freq:

		plt.bar(key, freq[key])
	plt.show()

quit()

for i in range(1, 11):
	M = data.data()
	M.addFile("../stimuli/giovanni/audio"+str(i)+".mid")
	print("audio"+str(i)+".mid", "pitch",len(M.getData("pitch")[0]))
	print("audio"+str(i)+".mid", "length",len(M.getData("length")[0]))
	print()


quit()
M = data.data()

#M.parse("dataBaseTest/")
M.parse("../stimuli/giovanni/")

#M.addFile("dataBaseTest/easy.mid")

M.print()

for elem in M.getData("pitch"):
	print(len(elem))


print()

for elem in M.getData("length"):
	print(len(elem))

M.plotScores()