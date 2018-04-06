import sys
import math
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib.animation import FuncAnimation
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
def cret(shape):
	return tf.Variable( tf.truncated_normal(shape, stddev=0.1))

####################################
# all the params you need to adjust#
####################################

epoch = 200000
neuron = [28*28,10,10,10,10]
learning_rate= 1e-4
saving_name = sys.argv[1]
bt = [100*i for i in range(1, 50)]
# training_batch = 1000

recorder=[]
for training_batch in bt:
# for learning_rate in [1e-2,1e-3,1e-4]:
	tf.reset_default_graph()	
	############ Model Preparing ################
	x = tf.placeholder( tf.float32 , [None , 28*28])
	y_ = tf.placeholder( tf.float32 , [None, 10])
	a = tf.placeholder(tf.float32 , 1)
	w_shape = [ [neuron[i] , neuron[i+1]] for i in range(len(neuron)-1)]
	b_shape = [ [neuron[i+1]]  for i in range(len(neuron)-1)]
	w = [ cret(ws) for ws in w_shape]
	b = [ cret(bs) for bs in b_shape]
		
	out = x
	for noww,nowb in zip(w,b):
		out = tf.nn.selu(tf.matmul(out,noww) + nowb) 
	y = out
	sp_y = tf.split(y,[1,1,1,1,1,1,1,1,1,1],axis=1)

	Jb_norm = []
	trainable = tf.trainable_variables();
	for yi in range(len(sp_y)):
		d = tf.gradients(sp_y[yi],x)
		Jb_norm.append(d)
	sensitivity = tf.reduce_mean(tf.square(tf.concat(Jb_norm,axis=2)))
	
	sess = tf.Session()

	# writer = tf.summary.FileWriter("/tmp/tensorflow/MNIST", sess.graph)
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits =y,labels =y_)
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	correct_pred = tf.equal( tf.argmax(y,1) , tf.argmax(y_,1) )
	acc = tf.reduce_mean( tf.cast(correct_pred , tf.float32))

	tf.global_variables_initializer().run(session=sess)

	trainN = len(mnist.train.labels)
	testN= len(mnist.test.labels)

	############## Start Parallel Training ##################
	early_stop_idx = 0
	early_stop = 700
	for _ in range(epoch):
		batch  = mnist.train.next_batch(training_batch)

		sess.run(train_step,feed_dict={x : batch[0],y_: batch[1],})
		if _ % 100 == 0:
			train_acc , train_loss , train_sensitivity = sess.run( (acc,loss,sensitivity),feed_dict={x : mnist.train.images ,y_: mnist.train.labels,})
			test_acc  , test_loss  , test_sensitivity= sess.run( (acc,loss,sensitivity),feed_dict={x : mnist.test.images,y_: mnist.test.labels})
			record = (training_batch, _,train_acc	,np.mean(train_loss)	,train_sensitivity,  test_acc 	,np.mean(test_loss) 	,test_sensitivity)
			if _ % 1000 == 0:
				print("batch_size:%5d,epoch:%d,\tacc:%.2f,\tloss:%.5f,\tsensitivity:%.8g\ttest_acc:%.2f,\ttest_loss:%.5f,\ttest_sensitivity:%.8g"%record)
			
			if early_stop > np.mean(test_loss):
				early_stop_idx = _
				early_stop = np.mean(test_loss)
			elif _ - early_stop_idx > 500:
				print('early_stopped.')
				break


	print("\x1B[31m",end='')
	train_acc , train_loss , train_sensitivity = sess.run( (acc,loss,sensitivity),feed_dict={x : mnist.train.images ,y_: mnist.train.labels,})
	test_acc  , test_loss  , test_sensitivity= sess.run( (acc,loss,sensitivity),feed_dict={x : mnist.test.images,y_: mnist.test.labels})
	record = (training_batch,train_acc	,np.mean(train_loss)	,train_sensitivity,  test_acc 	,np.mean(test_loss) 	,test_sensitivity)
	print("batch_size:%5d,\tacc:%.2f,\tloss:%.5f,\tsensitivity:%.8g\ttest_acc:%.2f,\ttest_loss:%.5f,\ttest_sensitivity:%.8g"%record)
	print("\x1B[0m",end='')
	
	recorder.append(record)

	np.save(saving_name,np.array(recorder))