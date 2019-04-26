import sys 
sys.path.append('../')

from idyom import longTermModel
from idyom import data
from idyom import score
from idyom import idyom

import numpy as np
import matplotlib.pyplot as plt

L = idyom.idyom(maxOrder=20, jump=True, maxDepth=10)

M = data.data(quantization=6)

#M.parse("../dataset/")
#M.parse("../datasetprout/")
M.parse("../examples/dataBaseTest")

L.train(M)

L.sample([{"pitch": 74, "length": 24}])

s = L.generate(500)

print(s.getData())

s.plot()

s.writeToMidi("exGen.mid")

#L.benchmarkQuantization("../dataset/",train=0.8)
ret = L.benchmarkOrder("../lisp/midis/", 24, train=0.8)
#print(ret)

quit()

S = L.getSurprisefromFile("dataBaseTest2/easy.mid", zero_padding=True)

plt.plot(S)
plt.xlabel("Time in quantization step")
plt.ylabel("Expected surprise (-log2(p))")
plt.show()

S = L.getSurprisefromFolder("dataBaseTest2/", zero_padding=True)

plt.plot(S[0])
plt.xlabel("Time in quantization step")
plt.ylabel("Expected surprise (-log2(p))")
plt.show()


quit()


#L.benchmarkQuantization("../dataset/",train=0.8)
ret = L.benchmarkOrder("../datasetprout/", 24, train=0.05)
#print(ret)

quit()



L.save("idyomModel.save")
