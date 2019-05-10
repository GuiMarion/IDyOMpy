import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime
import numpy as np


x1 = np.random.randint(10, size=100) +1
x2 = np.random.randint(10, size=100) +1


def plotLikelihoods(x1, X2):

	plt.title("Likelihoods over pieces")
	plt.xlabel("pieces")
	plt.ylabel("likelihood")
	ax = plt.subplot(111)

	for i in range(len(x1)):
		ax.bar(i-0.2, x1[i], width=0.2, color='b', align='center')
		ax.bar(i, x2[i], width=0.2, color='g', align='center')

	plt.show()

	plt.title("Likelihood diferences over pieces")
	plt.xlabel("pieces")
	plt.ylabel("likelihood diference (idyom - jump)")
	plt.plot(x1-x2)
	plt.show()

plotLikelihoods(x1, x2)