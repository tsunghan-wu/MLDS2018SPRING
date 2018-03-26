import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib.animation import FuncAnimation
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
####################################
# all the params you need to adjust#
####################################

choose_model = 1
epoch = 40000
if choose_model == 1:
	neuron = [28*28,15,15,10]
	learning_rate1 , learning_rate2 	= 1e-4 	, 1e-4
	training_batch1 , training_batch2 	= 1024	,	64
elif choose_model == 2: 
	neuron = [28*28,15,15,10]
	learning_rate1 , learning_rate2 	= 1e-4 	, 1e-3
	training_batch1 , training_batch2 	= 1024	, 1024
elif choose_model == 3: 
	neuron = [28*28,10,10,10,10]
	learning_rate1 , learning_rate2 	= 1e-4 	, 1e-4
	training_batch1 , training_batch2 	= 1024	,	64
elif choose_model == 4: 
	neuron = [28*28,100,10]
	learning_rate1 , learning_rate2 	= 1e-4 	, 1e-2
	training_batch1 , training_batch2 	= 1024	, 1024
else:
	neuron = [28*28,5,5,5,5,5,5,5,10]
	learning_rate1 , learning_rate2 	= 1e-3 	, 1e-2
	training_batch1 , training_batch2 	= 1024	, 1024
	epoch = 200000

saving_name = '1-3-3-1/record' + str(choose_model)


############ Model Preparing ################
x = tf.placeholder( tf.float32 , [None , 28*28])
y_ = tf.placeholder( tf.float32 , [None, 10])
a = tf.placeholder(tf.float32 , 1)
def cret(shape):
	return tf.Variable( tf.truncated_normal(shape, stddev=0.1))

def crep(shape):
	return tf.placeholder( tf.float32 , shape)

w_shape = [ [neuron[i] , neuron[i+1]] for i in range(len(neuron)-1)]
b_shape = [ [neuron[i+1]]  for i in range(len(neuron)-1)]
# p: perturb
w1 , b1 , w2 , b2 , wm , bm = [],[],[],[],[],[]
for ws , bs in zip(w_shape , b_shape):
	w1.append(cret(ws))
	w2.append(cret(ws))
	wm.append((1-a)*w1[-1] + a*w2[-1])
	b1.append(cret(bs))
	b2.append(cret(bs))
	bm.append((1-a)*b1[-1] + a*b2[-1])
	
out = x
for noww,nowb in zip(w1,b1):
	out = tf.nn.selu(tf.matmul(out,noww) + nowb) 
y1 = out
out = x
for noww,nowb in zip(w2,b2):
	out = tf.nn.selu(tf.matmul(out,noww) + nowb) 
y2 = out
out = x
for noww,nowb in zip(wm,bm):
	out = tf.nn.selu(tf.matmul(out,noww) + nowb) 
ym = out

sess = tf.Session()

loss_1 = tf.nn.softmax_cross_entropy_with_logits_v2(logits =y1,labels =y_)
train_step_1 = tf.train.AdamOptimizer(learning_rate1).minimize(loss_1)
correct_pred_1 = tf.equal( tf.argmax(y1,1) , tf.argmax(y_,1) )
acc_1 = tf.reduce_mean( tf.cast(correct_pred_1 , tf.float32))

loss_2 = tf.nn.softmax_cross_entropy_with_logits_v2(logits =y2,labels =y_)
train_step_2 = tf.train.AdamOptimizer(learning_rate2).minimize(loss_2)
correct_pred_2 = tf.equal( tf.argmax(y2,1) , tf.argmax(y_,1) )
acc_2 = tf.reduce_mean( tf.cast(correct_pred_2 , tf.float32))


loss_m = tf.nn.softmax_cross_entropy_with_logits_v2(logits =ym,labels =y_)
correct_pred_m = tf.equal( tf.argmax(ym,1) , tf.argmax(y_,1) )
acc_m = tf.reduce_mean( tf.cast(correct_pred_m , tf.float32))


tf.global_variables_initializer().run(session=sess)
############## Start Parallel Training ##################
for _ in range(epoch):
	batch1  = mnist.train.next_batch(training_batch1)
	batch2  = mnist.train.next_batch(training_batch2)

	sess.run(train_step_1,feed_dict={x : batch1[0],y_: batch1[1],})
	sess.run(train_step_2,feed_dict={x : batch2[0],y_: batch2[1],})
	if _ % 100 == 0:
		train_acc ,      loss = sess.run( (acc_1,loss_1),feed_dict={x : batch1[0],y_: batch1[1],})
		test_acc  , test_loss = sess.run( (acc_1,loss_1),feed_dict={x : mnist.test.images,y_: mnist.test.labels})
		print("(1)epoch:%d,\tacc:%.2f,\tloss:%.5f,\ttest_acc:%.2f,\ttest_loss:%.5f"%(_,train_acc,np.mean(loss ),test_acc,np.mean(test_loss)) ,end='\t')
		train_acc ,      loss = sess.run( (acc_2,loss_2),feed_dict={x : batch2[0],y_: batch2[1],})
		test_acc  , test_loss = sess.run( (acc_2,loss_2),feed_dict={x : mnist.test.images,y_: mnist.test.labels})
		print("(2)epoch:%d,\tacc:%.2f,\tloss:%.5f,\ttest_acc:%.2f,\ttest_loss:%.5f"%(_,train_acc,np.mean(loss ),test_acc,np.mean(test_loss)) )

############# Start move alpha ###########################
recorder = []
for rate in np.arange(-1,2,0.01):
	train_acc ,      loss = sess.run( (acc_m,loss_m),feed_dict={x : mnist.train.images,y_: mnist.train.labels, a : [rate]})
	test_acc  , test_loss = sess.run( (acc_m,loss_m),feed_dict={x : mnist.test.images ,y_: mnist.test.labels , a : [rate]})
	record_data = (rate,train_acc,np.mean(loss),test_acc,np.mean(test_loss))
	print("(m)rate:%.3f,\tacc:%.2f,\tloss:%.5f,\ttest_acc:%.2f,\ttest_loss:%.5f"%record_data)
	recorder.append(record_data)

np.save(saving_name,np.array(recorder))