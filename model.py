import numpy as np

SIZE = 10

def P(X, Y, z):
	ret = 0
	for value in X:
		ret += X[value] * Y[value]

	return ret

for i in range(100):
	X = np.random.rand(SIZE)
	X = X/np.sum(X)
	
	Y = np.random.rand(SIZE)
	Y = Y/np.sum(Y)

	print( 
