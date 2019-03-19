import sys 
sys.path.append('../')

from idyom import longTermModel
from idyom import data
from idyom import score

import numpy as np

L = longTermModel.longTermModel("pitch")

M = data.data()

M.parse("dataBaseTest/")

L.train(M.getData())

G = L.generate(500)

print(G)

s = score.score(G)

s.plot()

s.writeToMidi("exGen.mid")


L.save("longTermModel.save")
