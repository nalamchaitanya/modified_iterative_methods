import scipy.io as io

def loadMatrix(file):
	return io.mmread(file).toarray()