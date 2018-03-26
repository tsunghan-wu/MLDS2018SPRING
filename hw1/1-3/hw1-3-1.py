import tensorflow as tf
import numpy as np 
import models
import random
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.examples.tutorials.cifar10 import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#cifar10 = input_data.read_data_sets("CIFAR10_data/", one_hot=True)

#### global parameters #### 
learning_rate = 1e-4
tf.set_random_seed(7122)
###########################

x = tf.placeholder( tf.float32 , [ None , 28*28 ])
y_= tf.placeholder( tf.float32 , [ None , 10 ])
y = models.model(x)


#### model compile ####
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits =y,labels =y_)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_pred = tf.equal( tf.argmax(y,1) , tf.argmax(y_,1) )
acc = tf.reduce_mean( tf.cast(correct_pred , tf.float32))


# for tensorboard.
# writer = tf.summary.FileWriter("/tmp/tensorflow/MNIST", sess.graph)

for fake_or_true in ['fake']:
	print( '{0:-^40s}'.format(fake_or_true))
	if fake_or_true is 'fake':
		random.shuffle(mnist.train.labels)

	sess = tf.Session()
	tf.global_variables_initializer().run(session=sess)

	error_table = []
	for i in range(50000):
		batch = mnist.train.next_batch(1000)

		train_step.run(session = sess , feed_dict={
				x : batch[0],
				y_: batch[1],
			})
		if i%10 == 0:
			train_acc,loss = sess.run( (acc,cross_entropy),feed_dict={
					x : batch[0],
					y_: batch[1],
				})
			test_acc,test_loss = sess.run( (acc,cross_entropy),feed_dict={
				x : mnist.test.images,
				y_: mnist.test.labels,
			})
			error_table.append([i , train_acc , np.mean(loss), test_acc , np.mean(test_loss)])
			if i%100 == 0:
				print("step:%d,\tacc:%.2f,\tloss:%.5f,\ttest_acc:%.2f,\ttest_loss:%.5f"%(i,train_acc,np.mean(loss ),test_acc,np.mean(test_loss)) )

	saver = tf.train.Saver()
	import csv 
	if fake_or_true == 'fake':
		saver.save(sess,"fake_model/model.ckpt")
		cout = csv.writer(open('fake_loss.csv' , 'w'))
	else:
		saver.save(sess,"true_model/model.ckpt")
		cout = csv.writer(open('true_loss.csv' , 'w'))
	
	cout.writerows(error_table)

	sess.close()

############### Output Prepare ###################
