import sys 
sys.path.append('../')

from idyom import longTermModel
from idyom import data
from idyom import score
from idyom import idyom

import numpy as np
import matplotlib.pyplot as plt

L = idyom.idyom()

M = data.data()

M.parse("../dataset/")
#M.parse("dataBaseTest/")

L.train(M)

S = L.getLikelihoodfromFile("dataBaseTest2/easy.mid")

print(S)

S = L.getLikelihoodfromFolder("dataBaseTest2/")

print(S)

L.sample([{"pitch": 74, "length": 24}])

s = L.generate(500)

print(s.getData())

s.plot()

s.writeToMidi("exGen.mid")


L.save("idyomModel.save")
