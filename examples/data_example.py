import sys 
sys.path.append('../')

from idyom import data
import numpy as np



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