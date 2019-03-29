import sys 
sys.path.append('../')

from idyom import data
import numpy as np

M = data.data()

M.parse("dataBaseTest/")

M.addFile("dataBaseTest/easy.mid")

M.print()

for elem in M.getData("pitch"):
	print(elem)

print()

for elem in M.getData("length"):
	print(elem)