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
	num_pics = 40
	X, Y= load_pictures(num_pics)
	X = np.array(X)
	Y = np.array(Y)

	print X.shape
	print Y.shape

	# 0: pixel, 1: row, 2: picture
	mode = 2
	outstr = "Test score with data arranged by "

	if mode == 0:
		X, Y = arrange_by_pixel(X, Y)
		outstr += "pixel:"
	elif mode == 1:
		X, Y = arrange_by_row(X, Y)
		outstr += "row:"
	elif mode == 2:
		X, Y = arrange_by_picture(X, Y)
		outstr += "picture:"

	print X.shape
	print Y.shape

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
	print outstr
	print results

def arrange_by_pixel(X, Y):
	Xout = []
	Yout = []
	for image in X:
		for row in image:
			Xout.append(row)

	for image in Y:
		for row in image:
			Yout.append(row)

	for i, y in enumerate(Yout):
		for j, p in enumerate(y):
			if p == 255:
				Yout[i][j] = 2

	Xout = np.array(Xout)
	Yout = np.array(Yout)
	return Xout, Yout

def arrange_by_row(X, Y):
	Xout = []
	Yout = []
	for image in X:
		modified_image = []
		for row in image:
			flattened_row = np.hstack(row)
			modified_image.append(flattened_row)
		Xout.append(modified_image)

	for image in Y:
		modified_image = []
		for row in image:
			if 255 in row:
				modified_image.append(1)
			else:
				modified_image.append(0)
		Yout.append(modified_image)

	Xout = np.array(Xout)
	Yout = np.array(Yout)
	return Xout, Yout

def arrange_by_picture(X, Y):
	Xout = []
	Yout = []
	for image in X:
		Xout.append(np.hstack(image))
		#for row in image:
		#	Xout.append(row)

	for image in Y:
		modified_image = []
		for row in image:
			if 255 in row:
				modified_image.append(1)
			else:
				modified_image.append(0)
		Yout.append(modified_image)

	Xout = np.array(Xout)
	Yout = np.array(Yout)
	return Xout, Yout

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
