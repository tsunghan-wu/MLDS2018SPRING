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
def discriminator(inputs,batch_size,tag ,is_training = True):
	with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
		ensemble = []
		for i in range(4):
			dc0 = lrelu(conv2d(inputs, 32 ,name='d_conv0' +'_' +  str(i)))
			dc1 = lrelu(batch_norm('d_bn0'+'_' +str(i))(conv2d(dc0, 64 ,name='d_conv1'+'_' +str(i)),is_training))
			dc2 = lrelu(batch_norm('d_bn1'+'_' +str(i))(conv2d(dc1, 128 ,name='d_conv2'+'_' +str(i)),is_training))
			dc3 = lrelu(batch_norm('d_bn2'+'_' +str(i))(conv2d(dc2, 256 ,name='d_conv3'+'_' +str(i)),is_training))
			dc4 = lrelu(batch_norm('d_bn3'+'_' +str(i))(conv2d(dc3, 512 ,name='d_conv4'+'_' +str(i)),is_training))
			fc1 = tf.reshape(fully_connected('d_fc1'+'_' +str(i),tag,256) , [-1,1,1,256])
			tag_filter = tf.tile(fc1 , [1,4,4,1])
			dc4 = tf.concat([dc4,tag_filter],-1)
			dc5 = lrelu(batch_norm('d_bn4'+'_'+str(i))(conv2d(dc4, 512 , k_h=1, k_w=1, d_h=1, d_w=1,name='d_conv5'+'_' +  str(i)),is_training))
			flat = tf.reshape(dc5,[batch_size,-1])
			fc2 = fully_connected('d_fc2'+'_'+str(i),flat,1)
			ensemble.append(fc2)
		
		ensemble = tf.concat(ensemble,-1)
		voting = tf.reduce_sum(ensemble,0)																											
		return tf.sigmoid(voting),voting , ensemble

def generator(batch_size , input_tensor,tag, is_training = True):
	with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
		input_tensor = tf.random_normal([batch_size, 200])
		fc1 = lrelu(fully_connected('g_fc0',tag,16))
		input_tensor = tf.concat([input_tensor,fc1] , -1)

		h1 = fully_connected('g_fc_1', input_tensor, 4*4*64*8)
		c0 = tf.nn.relu(batch_norm('g_bn0')(tf.reshape(h1 , [-1,4,4,64*8]),is_training))
		c1 = tf.nn.relu(batch_norm('g_bn1')(deconv2d(c0 , [batch_size,8,8,64*4], name='g_deconv1'),is_training))
		c2 = tf.nn.relu(batch_norm('g_bn2')(deconv2d(c1 , [batch_size,16,16,128*2], name='g_deconv2'),is_training))
		c3 = tf.nn.relu(batch_norm('g_bn3')(deconv2d(c2 , [batch_size,32,32,256*1], name='g_deconv3'),is_training))
		c4 = tf.nn.relu(batch_norm('g_bn4')(deconv2d(c3 , [batch_size,64,64,128], name='g_deconv4'),is_training))
		c5 = tf.nn.relu(batch_norm('g_bn5')(deconv2d(c4 , [batch_size,128,128,64], name='g_deconv5'),is_training))
		c6 = conv2d(c5, 3 , k_h=1, k_w=1, d_h=1, d_w=1,name='conv6' , padding='VALID')
		return tf.nn.tanh(c6)

def graph(batch_size ,lr = 1e-5 , training_mode=True):
	global gen_train_op , dis_train_op 
	global gen_loss , dis_loss , fake_label , fake_dis
	global real_data , real_tag , fake_tag , perturbed_real_data
	global placeholder_is_training
	with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
		real_tag = tf.placeholder(tf.float32,shape=[batch_size,23])
		placeholder_is_training = tf.placeholder(tf.bool , shape=())
		with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
			fake_data = generator(batch_size , None , real_tag , placeholder_is_training)
			fake_score , fake_label ,fake_dis= discriminator(fake_data,batch_size , real_tag ,placeholder_is_training)

		if training_mode:
			real_data = tf.placeholder(tf.float32, shape=[batch_size,128, 128, 3])
			perturbed_real_data = real_data + 0.5 * tf.keras.backend.std(real_data,0) * tf.random_uniform(real_data.get_shape())
			fake_tag = tf.placeholder(tf.float32,shape=[batch_size,23])
			with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
				fake_data = generator(batch_size , None , real_tag , placeholder_is_training)
				real_score , real_label ,real_dis= discriminator(real_data,batch_size , real_tag ,placeholder_is_training)
			
			d_vars = tf.trainable_variables('discriminator')
			g_vars = tf.trainable_variables('generator')
			# loss functrion

			gen_loss = -tf.reduce_mean(fake_label)
			dis_loss = tf.reduce_mean(fake_label) - tf.reduce_mean(real_label)
			
			# WGAN-GP
			## 1. interpolation
			alpha = tf.random_uniform(shape=[batch_size],minval=0.,maxval=1.)
			img_size = 128
			# def slerp(low, high):
			# 	def dot(A,B):
			# 		return tf.reduce_sum(tf.multiply(A,B),[1,2,3])
			# 	dot_value = dot(low/tf.norm(low), high/tf.norm(high))
			# 	omega = tf.acos(dot_value)
			# 	so = tf.sin(omega)
			# 	w1 =   tf.reshape(tf.sin((1.0-alpha)*omega) / so ,[batch_size,1,1,1])
			# 	w2 =   tf.reshape(tf.sin(alpha*omega) / so ,[batch_size,1,1,1])
			# 	return w1*low + w2*high
			# interpolation= slerp(fake_data , real_data)
			# ## 2. gradient penalty
			# gradients = tf.gradients(discriminator(interpolation,batch_size,real_tag , placeholder_is_training)[1], [interpolation])[0]
			# slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
			# gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
			# ## 3. append it to loss function
			# dis_loss += (20 * gradient_penalty)
			
			with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
				gen_train_op = tf.train.AdamOptimizer(
					learning_rate=lr,beta1=0.5).minimize(gen_loss,var_list=g_vars)
				dis_train_op = tf.train.AdamOptimizer(
					learning_rate=lr,beta1=0.5).minimize(dis_loss,var_list=d_vars)

def train(danager ,start_epoch = 61,batch_size=32, flog=sys.stdout):
	EPOCH = 240
	iterations = 10

	for epoch in range(start_epoch,EPOCH):
		for iters in range(iterations):
			# get next batch
			for dis_iter in range(40):
				img, tag1 = danager.get_batch()
				_, d_loss = sess.run([dis_train_op, dis_loss], feed_dict={real_data: img , real_tag: tag1 , placeholder_is_training:True})
			for gen_iter in range(40):
				img, tag1 = danager.get_batch()
				_, g_loss = sess.run([gen_train_op, gen_loss], feed_dict={real_tag: tag1, placeholder_is_training:True})
			def brief(f):
				return "{:e}".format(f)[:6] + "{:e}".format(f)[-3:]
			
			print ("epoch: {}, iter: {:04d}, dis_loss:{} , gen_loss:{} ".format(epoch,iters , brief(d_loss), brief(g_loss)), file=flog)
			# print ("epoch: {}, iter: {:04d}, gen_loss:{} ".format(epoch,iters ,  brief(g_loss)), file=flog)
			# print ("epoch: {}, iter: {:04d}, dis_loss:{}".format(epoch,iters , brief(d_loss)), file=flog)
			flog.flush(	)
		# sample per 100 iterations
		with tf.variable_scope(tf.get_variable_scope()):
			samples = generator(batch_size , None , real_tag)
			gen_imgs = sess.run(samples , feed_dict={real_tag:tag1 , placeholder_is_training:False})
			# np.save("epoch"+str(epoch)+"output.npy", gen_imgs)
			# regular_result(self , gen_imgs ,tag = None, save_file = None  , r=4,c=6):
			danager.regular_result(gen_imgs , tag1 , save_file='result/result' + str(epoch) + '.jpg')
			# save_imgs(gen_imgs, epoch , tag1)
			
		# save model per 100 iterations
		if epoch % 1 == 0:
			checkpoint_path = "./epoch"+str(epoch)+"/wgan_gp.ckpt"
			saver.save(sess, checkpoint_path)
					
			
def testing(batch_size ,tag_file = 'my_testing_tags.txt'):
	fin = open(tag_file , 'r')
	hair_dict = {'orange': 0, 'white': 1, 'aqua': 2, 'grey': 3, 
            'green': 4, 'red': 5, 'purple': 6, 'pink': 7,
            'blue': 8, 'black': 9, 'brown': 10, 'blonde': 11,
            'gray': 12}
	eyes_dict = {'black': 0, 'orange': 1, 'pink': 2, 'yellow': 3, 
	            'aqua': 4, 'purple': 5, 'green': 6, 'brown': 7,
	            'red': 8, 'blue': 9}
	all_output = []
	for now_itr , row in enumerate(fin):
		row = row.split(',')[1].split()
		print('now process:' , now_itr , row)
		this_tag = np.zeros(23)
		this_tag[hair_dict[row[0]]] = 1
		this_tag[eyes_dict[row[2]] + 13] = 1
		this_tag = np.tile(this_tag,[batch_size,1])
		samples = generator(batch_size , None , real_tag , placeholder_is_training)

		all_img , all_score = [], []
		for _ in range(3):
			gen_imgs , score = \
				sess.run([samples,fake_dis], 
					feed_dict={real_tag:this_tag,placeholder_is_training:False })
			all_img.append(gen_imgs)
			all_score.append(score)
		gen_imgs = np.concatenate(all_img,0)
		score = np.concatenate(all_score,0)
		total_score  = np.sum ((score - np.mean(score,0).reshape(1,4)) / np.std(score,0).reshape(1,4) , 1)
		
		chosen = np.argsort(total_score)[::-1][:50]
		all_output.append(gen_imgs[chosen])
	np.save('tmp_result.npy' , all_output)
	
if __name__ == '__main__':
	seed = 8989
	if len(sys.argv) == 3:
		seed = sys.argv[2]
	tf.set_random_seed(seed)
	np.random.seed(seed)

	mode = 'testing'
	with tf.Session() as sess:
		if mode == 'training':	
			batch_size = 32
			danager = Danager(batch_size , workers=8)
			
			graph(batch_size , 1e-5 , training_mode=True)
			print('graph bulid up.')
			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())
			saver.restore(sess, "./epoch92/wgan_gp.ckpt")
			
			
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