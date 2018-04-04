import tensorflow as tf
import numpy as np 
import models_132
import random
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.examples.tutorials.cifar10 import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#cifar10 = input_data.read_data_sets("CIFAR10_data/", one_hot=True)

#### global parameters #### 
tf.set_random_seed(7122)
###########################



# for tensorboard.
# writer = tf.summary.FileWriter("/tmp/tensorflow/MNIST", sess.graph)
recorder = []
for time_stamp in range(100):
	tf.reset_default_graph()

	print( '{0:-^40s}'.format(str(time_stamp) + 'times'))

	x = tf.placeholder( tf.float32 , [ None , 28*28 ])
	y_= tf.placeholder( tf.float32 , [ None , 10 ])
	#### local random variable ####
	rrr = np.random.randint(1,5)
	rr = lambda : np.random.randint(rrr,rrr+3)
	layers = np.random.randint(2,4)
	par = [ rr() for _ in range(layers)]
	y = models_132.model(x,params=par + [10])
	learning_rate = np.random.uniform(1e-3,1e-5)
	###############################


	#### model compile ####
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits =y,labels =y_)
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

	correct_pred = tf.equal( tf.argmax(y,1) , tf.argmax(y_,1) )
	acc = tf.reduce_mean( tf.cast(correct_pred , tf.float32))


	sess = tf.Session()
	tf.global_variables_initializer().run(session=sess)
	all_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
	print("\x1B[31m",end='')
	print('all_params' , all_params)
	print("\x1B[0m",end='')
	error_table = []
	for i in range(5000):
		batch = mnist.train.next_batch(1000)

		train_step.run(session = sess , feed_dict={
				x : batch[0],
				y_: batch[1],
			})
		if i%500 == 0:
			train_acc,loss = sess.run( (acc,cross_entropy),feed_dict={
					x : batch[0],
					y_: batch[1],
				})
			test_acc,test_loss = sess.run( (acc,cross_entropy),feed_dict={
				x : mnist.test.images,
				y_: mnist.test.labels,
			})
			print("step:%d,\tacc:%.2f,\tloss:%.5f,\ttest_acc:%.2f,\ttest_loss:%.5f"%(i,train_acc,np.mean(loss ),test_acc,np.mean(test_loss)) )
	
	train_acc,loss = sess.run( (acc,cross_entropy),feed_dict={
			x : mnist.train.images,
			y_: mnist.train.labels,
		})
	test_acc,test_loss = sess.run( (acc,cross_entropy),feed_dict={
		x : mnist.test.images,
		y_: mnist.test.labels,
	})
	this_record = (all_params , train_acc , np.mean(loss), test_acc,np.mean(test_loss))
	print("\x1B[31m",end='')
	print("all_params:%d,\tacc:%.2f,\tloss:%.5f,\ttest_acc:%.2f,\ttest_loss:%.5f"%this_record)
	print("\x1B[0m",end='')
	recorder.append(this_record)

# import csv 
# cout = csv.writer(open('1-3-2_few_v4.csv' , 'w'))
# cout.writerows(recorder)

############### Output Prepare ###################
