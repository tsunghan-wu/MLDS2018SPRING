import numpy as np
import keras
from keras import initializers
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from extradata_process import Danager
import tensorflow as tf

tf.set_random_seed(777)
np.random.seed(777)

hair_dict = {'orange': 0, 'white': 1, 'aqua': 2, 'grey': 3, 
        'green': 4, 'red': 5, 'purple': 6, 'pink': 7,
        'blue': 8, 'black': 9, 'brown': 10, 'blonde': 11,
        'gray': 12}
eyes_dict = {'black': 0, 'orange': 1, 'pink': 2, 'yellow': 3, 
            'aqua': 4, 'purple': 5, 'green': 6, 'brown': 7,
            'red': 8, 'blue': 9}


model = load_model('3_2/model.hdf5')
### COMPILE ###
model.compile(loss='binary_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])
checkpoint = ModelCheckpoint('model.hdf5', monitor='acc', verbose=1, save_best_only=True)

def softmax(x):
	"""Compute softmax values for each sets of scores in x."""
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()

import sys
allX = np.load('tmp_result.npy')
fin = open(sys.argv[1], 'r')
all_tags = []
final_result = []
for X,(now_itr , row) in zip(allX,enumerate(fin)):
	raw = row
	row = row.split(',')[1].split()
	hair_target = hair_dict[row[0]]
	eyes_target = eyes_dict[row[2]]
	
	this_tag = np.zeros(23)
	this_tag[hair_dict[row[0]]] = 1
	this_tag[eyes_dict[row[2]] + 13] = 1
	all_tags.append(this_tag)


	# print(hair_target , eye_target)	
	y = model.predict(X)
	choice = []
	for idx,(i,now_y) in enumerate(zip(X,y)):
		hair = now_y[:13]
		eyes = now_y[13:]
		color_score = hair[hair_target] + eyes[eyes_target]
		# print(color_score)
		choice.append( [color_score , idx] )	
	
	A = np.argsort(np.array(choice)[:,0])[::-1]
	ok = X[A[0]]

	print(raw[:-1],'best_score :' , choice[A[0]])
	if ok is None:	
		final_result.append(X[0])
	else:
		final_result.append(ok)

Danager.regular_result(final_result ,tag =None, save_file = 'samples/cgan.png',r=5,c=5)