import sys 
sys.path.append('../')

from idyom import longTermModel
import numpy as np

L = longTermModel.longTermModel("pitch", 10)

X = np.arange(1000) % 10

L.train(X)


