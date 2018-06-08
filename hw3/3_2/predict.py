import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from skimage import io
from skimage import transform
from util import *
from extradata_process import Danager

tf.set_random_seed(777)
np.random.seed(777)


batch_size = 50
real_data = tf.placeholder(tf.float32, shape=[batch_size,128, 128, 3])
placeholder_is_training = tf.placeholder(tf.bool , shape=())
real_tag = tf.placeholder(tf.float32,shape=[batch_size,23])

i = 0
dc0 = lrelu(conv2d(real_data, 32 ,name='d_conv0' +'_' +  str(i)))
dc1 = lrelu(batch_norm('d_bn0'+'_' +str(i))(conv2d(dc0, 64 ,name='d_conv1'+'_' +str(i)),placeholder_is_training))
dc2 = lrelu(batch_norm('d_bn1'+'_' +str(i))(conv2d(dc1, 128 ,name='d_conv2'+'_' +str(i)),placeholder_is_training))
dc3 = lrelu(batch_norm('d_bn2'+'_' +str(i))(conv2d(dc2, 256 ,name='d_conv3'+'_' +str(i)),placeholder_is_training))
dc4 = lrelu(batch_norm('d_bn3'+'_' +str(i))(conv2d(dc3, 512 ,name='d_conv4'+'_' +str(i)),placeholder_is_training))

flat = tf.reshape(dc4,[batch_size,-1])
fc2 = fully_connected('d_fc2'+'_'+str(i),flat,23)

hair_p = fc2[:,:13]
eyes_p = fc2[:,13:]
hair_r = real_tag[:,:13]
eyes_r = real_tag[:,13:]
hair_score = tf.nn.softmax(hair_p)
eyes_score = tf.nn.softmax(eyes_p)
hair_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = hair_r , logits=hair_p))
eyes_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = eyes_r , logits=eyes_p))
loss=  hair_loss + eyes_loss
train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5).minimize(loss)


EPOCH = 240
iterations = 1000

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
print('start to restore')
saver.restore(sess, "./3_2/dis2/wgan_gp.ckpt")




hair_dict = {'orange': 0, 'white': 1, 'aqua': 2, 'grey': 3, 
        'green': 4, 'red': 5, 'purple': 6, 'pink': 7,
        'blue': 8, 'black': 9, 'brown': 10, 'blonde': 11,
        'gray': 12}
eyes_dict = {'black': 0, 'orange': 1, 'pink': 2, 'yellow': 3, 
            'aqua': 4, 'purple': 5, 'green': 6, 'brown': 7,
            'red': 8, 'blue': 9}



import sys
allX = np.load('./tmp_result.npy')
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
	hh , ee = sess.run([hair_score , eyes_score] , feed_dict={real_data : X , placeholder_is_training : False})
	y = np.concatenate([hh,ee],-1)
	print(y.shape)
	# y = predict(X)
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

Danager.regular_result(final_result ,tag =None, save_file = './samples/cgan.png',r=5,c=5)