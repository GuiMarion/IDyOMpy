import sys 
sys.path.append('../')

from idyom import score
import matplotlib.pyplot as plt
import numpy as np
import copy

# Import one of my masterpieces ...
#s = score.score("velocity.mid")

s = score.score("../stimuli/Chinese/test1/chinese-001.mid")
s.plot()

print(s.getData())


quit()

s = score.score("dataBaseTest/easy.mid")

s = score.score("../stimuli/giovanni/audio1.mid")


print(list(s.getData()))

tmp = list(s.getData())

for i in range(1, len(tmp)):
	if tmp[i] == -1:
		tmp[i] = tmp[i-1]

print(tmp)
plt.plot(tmp)
plt.show()

quit()

s.fromData(s.getData())

s.plot()

s.writeToMidi("dataBaseTest/out.mid")

quit()

'''
Part for the midi
'''

# Return the length in time beat
print("length:", s.getLength())

# Plot the piano roll representation of the score
s.plot()

# print the pianoRoll matrix
print(s.getPianoRoll())


# Extract all parts of size 1 beat with a step of 10 samples
L = s.extractAllParts(1, step= 10000)
for elem in L:
	elem.plot()


'''
Part for the extractPart
'''


# plot the 10th first beats
sub = s.extractPart(0, 10)
sub.plot()

# print the pianoroll matrix
print(sub.getPianoRoll())

# Return the length in time beat
print("length:", sub.getLength())


T = sub.getTransposed()
for elem in T:
	elem.plot()