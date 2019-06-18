import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../../../")
from idyom import data
sys.path.append("stimuli/giovanni/surprises/")

SERVER = False

folder = "../../../figs/server/stimuli/"

def compareLikelihoods(x1, x2, name="kikou.eps"):

	plt.title("Likelihoods over pieces")
	plt.xlabel("pieces")
	plt.ylabel("likelihood")
	ax = plt.subplot(111)

	for i in range(len(x1)):
		ax.bar(i-0.2, x1[i], width=0.2, color='b', align='center')
		ax.bar(i, x2[i], width=0.2, color='g', align='center')

	plt.savefig(name+"_1.eps")
	plt.show()

	plt.title("Likelihood diferences over pieces")
	plt.xlabel("pieces")
	plt.ylabel("likelihood diference (idyom - jump)")
	plt.plot(np.array(x1)-np.array(x2))
	plt.plot(np.zeros(len(x1)))

	plt.savefig(name+"_2.eps")
	plt.show()

file1 = "surpriseSignal_lisp.pickle"
file2 = "surpriseSignal_bach_Pearce_jump_False_relativeEntropy.pickle"
file3 = "surpriseSignal_bach_Pearce_jump_True.pickle"

l1 = pickle.load( open(file1, "rb" ) )
l2 = pickle.load( open(file2, "rb" ) )
l3 = pickle.load( open(file3, "rb" ) )

likelihoods1 = []
likelihoods2 = []
likelihoods3 = []


for file in l1:

	l3[file] = np.nan_to_num(l3[file])
	l1[file] = np.nan_to_num(l1[file])

	likelihoods1.append(np.mean(l1[file]))
	likelihoods2.append(2**-np.mean(l2[file]))
	likelihoods3.append(2**-np.mean(l3[file]))

plt.bar([1,2,3], [np.mean(likelihoods1), np.mean(likelihoods2), np.mean(likelihoods3)], yerr=[np.std(likelihoods1), np.std(likelihoods2), np.std(likelihoods3)])
plt.savefig(folder+"comparisons.eps")
plt.show()

compareLikelihoods(likelihoods2, likelihoods1, name=folder+"compareLikelihoodsIDyOMpy_VS_IDyOM")

# ploting in the music space

M = data.data()
M.parse("../", augment=False)

dat2, files4 = M.getScoresFeatures()

weights = []
colors = []

for file in range(len(likelihoods1)):
	weights.append(80000*abs(likelihoods1[file]-likelihoods2[file])**2)
	if likelihoods1[file]-likelihoods2[file] < 0:
		colors.append('coral')
	elif likelihoods1[file]-likelihoods2[file] > 0:
		colors.append('deepskyblue')
	else:
		colors.append('black')


plt.scatter(dat2[0][:len(dat2[1])],dat2[1], s=weights, c=colors)

plt.title('IDyOMpy - IDyOM')
plt.xlabel('Average 1-note interval')
plt.ylabel('Average note onset')

plt.savefig(folder+"scoreSpaceIDyOMpy_VS_IDyOM.eps")
plt.show()


		# IDyOMpy VS JUMP


compareLikelihoods(likelihoods2, likelihoods3, name=folder+"compareLikelihoodsIDyOMpy_VS_Jump")

# ploting in the music space

M = data.data()
M.parse("../", augment=False)

dat2, files4 = M.getScoresFeatures()

weights = []
colors = []

for file in range(len(likelihoods3)):
	weights.append(80000*abs(likelihoods3[file]-likelihoods2[file])**2)
	if likelihoods3[file]-likelihoods2[file] < 0:
		colors.append('coral')
	elif likelihoods3[file]-likelihoods2[file] > 0:
		colors.append('deepskyblue')
	else:
		colors.append('black')


plt.scatter(dat2[0][:len(dat2[1])],dat2[1], s=weights, c=colors)

plt.title('IDyOMpy - Jump')
plt.xlabel('Average 1-note interval')
plt.ylabel('Average note onset')

plt.savefig(folder+"scoreSpaceIDyOMpy_VS_Jump.eps")
plt.show()

