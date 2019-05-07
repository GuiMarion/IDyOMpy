from idyom import jumpModel

import numpy as np
J = jumpModel.jumpModel("pitch", maxOrder=10, maxDepth=10)
X = np.arange(1000) % 10
print(X)

J.train(X)

state = [1,2,3,4,5,6,7,8]

print(J.getPrediction(state))
