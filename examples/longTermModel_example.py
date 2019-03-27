import sys 
sys.path.append('../')

from idyom import longTermModel
from idyom import data
from idyom import score

import numpy as np
import matplotlib.pyplot as plt

L = longTermModel.longTermModel("pitch", maxOrder=20)

M = data.data()

M.parse("../dataset/")
#M.parse("dataBaseTest/")


L.train(M.getData())

G = L.generate(500)

print(G)

s = score.score(G)

s.plot()

s.writeToMidi("exGen.mid")


L.save("longTermModel.save")
