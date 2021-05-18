import sys 
sys.path.append('../')

from idyom import markovChain
from idyom import data
from idyom import score

import numpy as np

M = markovChain.markovChain(3)

D = data.data()
D.parse("dataBaseTest/")
M.train(D.getData("pitch"))

print(D.getData("pitch"))


S = M.generate(500)

S.writeToMidi("generation1.mid")

S.toWaveForm("generation1.wav")

print(S.getData())


quit()

matrix = M.getStatesMatrix()
print(M.transitions)

print(matrix)

print(M.probabilities)
matrix2 = M.getMatrix()
print(matrix2)

M = markovChain.markovChain(2)

X = np.arange(10000) % 10

M.train(X)

matrix = M.getStatesMatrix()
print(M.transitions)

print(matrix)

print(M.probabilities)
matrix2 = M.getMatrix()
print(matrix2)

print(M.getPrediction("[1, 2]"))

print(M.getLikelihood("[1, 2]", '3'))

M.save("markovChain.mc")