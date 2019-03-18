import sys 
sys.path.append('../')

from idyom import score
import matplotlib.pyplot as plt
import numpy as np
import copy

# Import one of my masterpieces ...
#s = score.score("velocity.mid")
s = score.score("dataBaseTest/easy.mid")

print(s.getData())

s.fromData(s.getData())

s.plot()

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