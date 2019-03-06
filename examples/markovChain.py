import sys 
sys.path.append('../')

from idyom import markovChain
import numpy as np

M = markovChain.markovChain(1)

X = np.arange(10000000) % 10



M.train(X)

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