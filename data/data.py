"""
Loads the data from Matrix Market
"""
import scipy.io as io

def loadMatrix(file):
	""" Reads the .mtx file downloaded from Matrix Market and converts it to numpy ndarray. """
	return io.mmread(file).toarray()