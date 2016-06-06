import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
sys.path.append('/Library/Python/2.7/site-packages')
import cvxopt
import math
from pystruct.models import ChainCRF
from pystruct.learners import FrankWolfeSSVM


def chaincrf_test():
	num_pics = 9
	X, Y= load_pictures(num_pics)
	X = np.array(X[0])
	Y = np.array(Y[0])
	for i, y in enumerate(Y):
		for j, p in enumerate(y):
			if p == 255:
				Y[i][j] = 2
    

	#print X.shape, Y.shape
	train_pct = 0.66
	test_pct = 1 - train_pct
	X_train = X[0:math.floor(train_pct * num_pics)]
	X_test = X[math.floor(test_pct*num_pics):]
	Y_train = Y[0:math.floor(train_pct * num_pics)]
	Y_test = Y[math.floor(test_pct*num_pics):]

	model = ChainCRF()
	ssvm = FrankWolfeSSVM(model=model, C=.1, max_iter=10)
	# #print X_train.shape, Y_train.shape
	ssvm.fit(X_train, Y_train)
	results = ssvm.score(X_test, Y_test)
	print results


def load_pictures(num):
	array_X = []
	array_Y = []
	offset = 16092
	for i in range(offset, offset+num):
		img_file = "imagery/image_city_" + str(i) + ".tif"		
		bldg_file = "buildings/bldg_city_" + str(i) + ".tif"
		I = plt.imread(img_file)
		J = plt.imread(bldg_file)
		I = np.resize(I, (127, 127, 4))
		J = np.resize(J, (127, 127))
		array_X.append(I)
		array_Y.append(J)
	return array_X, array_Y


chaincrf_test()
