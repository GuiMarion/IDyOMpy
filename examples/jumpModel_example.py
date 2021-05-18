import sys 
sys.path.append('../')

from idyom import jumpModel
from idyom import data
from idyom import score

import numpy as np
import matplotlib.pyplot as plt

L = jumpModel.jumpModel("pitch", maxDepth=10, maxOrder=20)

M = data.data()

M.parse("../datasetprout/")
#M.parse("dataBaseTest/")

L.train(M.getData("pitch"))

G = L.generate(500)

print(G)

s = score.score(G)

s.plot()

s.writeToMidi("exGen.mid")


L.save("jumpModel.save")
