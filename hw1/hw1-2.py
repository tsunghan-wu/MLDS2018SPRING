import tensorflow as tf
import numpy as np 
import models
from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.examples.tutorials.cifar10 import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#cifar10 = input_data.read_data_sets("CIFAR10_data/", one_hot=True)

#### global parameters #### 
learning_rate = 1e-4
###########################

x = tf.placeholder( tf.float32 , [ None , 28*28 ])
y_= tf.placeholder( tf.float32 , [ None , 10 ])
y = models.model1(x)


#### model compile ####
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits =y,labels =y_)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_pred = tf.equal( tf.argmax(y,1) , tf.argmax(y_,1) )
acc = tf.reduce_mean( tf.cast(correct_pred , tf.float32))

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

# for tensorboard.
# writer = tf.summary.FileWriter("/tmp/tensorflow/MNIST", sess.graph)

error_table = []
for i in range(30000):
	batch = mnist.train.next_batch(100)
	train_step.run(session = sess , feed_dict={
			x : batch[0],
			y_: batch[1],
		})
	train_acc,loss = sess.run( (acc,cross_entropy),feed_dict={
		x : batch[0],
		y_: batch[1],
	})
	error_table.append([i , train_acc , np.mean(loss)])
	if i%100 == 0:
		print("step:%d,\tacc:%g,\tloss:%g" % (i,train_acc,np.mean(loss)))

saver = tf.train.Saver()
saver.save(sess,"cnn2_dim32/model.ckpt")

############### Output Prepare ###################
import csv 
cout = csv.writer(open('error_table_cnn2_dim32.csv' , 'w'))
cout.writerows(error_table)
