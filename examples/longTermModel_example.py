import sys 
sys.path.append('../')

from idyom import longTermModel
from idyom import data
from idyom import score

import numpy as np

L = longTermModel.longTermModel("pitch", 10)

M = data.data()

M.parse("dataBaseTest/")

L.train(M.getData())


G = L.generate(len(M.getData()[0]))

print(G)

s = score.score(G)

s.plot()


L.save("longTermModel.save")
