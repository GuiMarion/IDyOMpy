import sys 
sys.path.append('../')

from idyom import data
import numpy as np

M = data.data()

M.parse("dataBaseTest/")

M.addFile("dataBaseTest/easy.mid")

M.print()
