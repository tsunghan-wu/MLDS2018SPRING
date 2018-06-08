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
hair_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = hair_r , logits=hair_p))
eyes_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = eyes_r , logits=eyes_p))
loss=  hair_loss + eyes_loss
train_op = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.5).minimize(loss)


EPOCH = 240
iterations = 100

def train(danager ,start_epoch = 61,batch_size=32, flog=sys.stdout):
	for epoch in range(start_epoch,EPOCH):
		for _ in range(iterations):
			img, tag1 = danager.get_batch()
			_, d_loss , H , E = sess.run([train_op, loss , hair_p ,eyes_p], feed_dict={real_data: img , real_tag: tag1 , placeholder_is_training:True})
			H = np.argmax(H,1)
			E = np.argmax(E,1)
			PH = np.argmax(tag1[:,:13] , 1)
			PE = np.argmax(tag1[:,13:] , 1)
		print( (np.sum(H==PH) + np.sum(E==PE))/100 , 'loss :' , d_loss )
		if epoch % 1 == 0:
			checkpoint_path = "./dis2/wgan_gp.ckpt"
			saver.save(sess, checkpoint_path)
					
	
if __name__ == '__main__':
	seed = 8989
	if len(sys.argv) == 3:
		seed = sys.argv[2]
	tf.set_random_seed(seed)
	np.random.seed(seed)

	mode = 'training'
	with tf.Session() as sess:
		if mode == 'training':	
			batch_size = 50
			danager = Danager(batch_size , workers=16)
			
			print('graph bulid up.')
			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())
			
			
			train(danager,start_epoch =93,batch_size=batch_size, flog=sys.stdout)
		else:
			batch_size = 128
			graph(batch_size , 2e-4 , training_mode=False)
			print('graph bulid up.')
			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())
			print('start to restore')
			saver.restore(sess, "./epoch82/wgan_gp.ckpt")
			testing(batch_size , sys.argv[1])