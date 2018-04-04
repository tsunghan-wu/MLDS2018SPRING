import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

############ Model Preparing ################
x = tf.placeholder( tf.float32 , [None , 28*28])
y_ = tf.placeholder( tf.float32 , [None, 10])

def cret(shape):
	return tf.Variable( tf.truncated_normal(shape, stddev=0.1))

def crep(shape):
	return tf.placeholder( tf.float32 , shape)

neuron = [784,10,10,10]
w_shape = [ [neuron[i] , neuron[i+1]] for i in range(len(neuron)-1)]
b_shape = [ [neuron[i+1]]  for i in range(len(neuron)-1)]
# p: perturb
w , b , pw , pb = [],[],[],[]
for ws , bs in zip(w_shape , b_shape):
	temp_w = cret(ws)
	temp_b = cret(bs)
	now_pw =crep(ws)
	now_pb =crep(bs)
	w.append(temp_w + now_pw)
	b.append(temp_b + now_pb)
	pw.append(now_pw)
	pb.append(now_pb)

rr = lambda size : np.random.uniform(-0.1,0.1,size) if np.random.randint(0,10) == 0 else np.zeros(size)

zero_w_dict = { now_pw: np.zeros(ws) for now_pw , ws in zip(pw,w_shape)}
zero_b_dict = { now_pb: np.zeros(bs) for now_pb , bs in zip(pb,b_shape)}
zero_dict = {}
zero_dict.update(zero_w_dict)
zero_dict.update(zero_b_dict)
out = x
for noww,nowb in zip(w,b):
	out = tf.nn.selu(tf.matmul(out,noww) + nowb)
y =  out
 
sess = tf.Session()
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits =y,labels =y_)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_pred = tf.equal( tf.argmax(y,1) , tf.argmax(y_,1) )
acc = tf.reduce_mean( tf.cast(correct_pred , tf.float32))
fout = open('recorder' , 'a')
tf.global_variables_initializer().run(session=sess)
############## Start Training ##################
parameters_table = []
loss_table = []
color_table = []
N = len(mnist.train.images)
epoch = 1000
recorder = []

for batch_size in range(100,4000,100):
	sess = tf.Session()
	tf.global_variables_initializer().run(session=sess)

	for _ in range(epoch):
		for __ in range(N//batch_size+1):
			trainX , trainY = mnist.train.next_batch(batch_size)
			now_feed_dict = {	x : trainX,	y_ : trainY	}
			now_feed_dict.update(zero_dict)
			sess.run(train_step,feed_dict=now_feed_dict)
		if _ % 10 == 0 :
			# random sampling
			ob_w, ob_b , eva_loss,eva_acc = sess.run([w,b,loss,acc],feed_dict=now_feed_dict)
			ob_w_flat = np.concatenate(np.array([ now_ob_w.flatten() for now_ob_w in ob_w ]),axis=0)
			ob_b_flat = np.concatenate(np.array([ now_ob_b.flatten() for now_ob_b in ob_b ]),axis=0)
			ob_flat = np.concatenate( [ob_w_flat,ob_b_flat],axis=0)
			parameters_table.append(ob_flat)
			loss_table.append(eva_loss)
			color_table.append(1)
			print ("epoch %d loss %8g , acc %4g" %(_,np.mean(eva_loss),eva_acc ))

	loss_table = []
	acc_table = []
	for __ in range(10000):
		pw_dict = { now_pw: rr(ws) for now_pw , ws in zip(pw,w_shape)}
		pb_dict = { now_pb: rr(bs) for now_pb , bs in zip(pb,b_shape)}
		p_dict = {}
		p_dict.update(pw_dict)
		p_dict.update(pb_dict)
		p_dict.update({x:trainX, y_:trainY})
		eva_loss , eva_acc = sess.run([loss , acc],feed_dict=p_dict)
		p_dict.update({x:mnist.test.images, y_:mnist.test.labels})
		loss_table.append(np.mean(eva_loss))
		acc_table.append(np.mean(eva_acc))

	rand_loss = np.mean(loss_table)
	rand_acc = np.mean(acc_table)

	now_feed_dict = {x : mnist.train.images,y_ : mnist.train.labels	}
	now_feed_dict.update(zero_dict)
	train_loss , train_acc = sess.run([loss,acc],feed_dict=now_feed_dict)
	now_feed_dict = {x : mnist.test.images,y_ : mnist.test.labels	}
	now_feed_dict.update(zero_dict)
	test_loss , test_acc = sess.run([loss,acc],feed_dict=now_feed_dict)

	print("\x1B[31m",end='')
	print("batch_size:" , batch_size , 
		"random loss:" ,  rand_loss , "random acc:" , rand_acc,
		"train loss:" , np.mean(train_loss) , "train acc:" , train_acc,
		"test loss:" , np.mean(test_loss) , "test acc:" , test_acc)
	print("batch_size:" , batch_size , 
		",random loss:" ,  rand_loss , ",random acc:" , rand_acc,
		",train loss:" , np.mean(train_loss) , ",train acc:" , train_acc,
		",test locdss:" , np.mean(test_loss) , ",test acc:" , test_acc , file=fout)
	print("\x1B[0m",end='')
	fout.flush()
	sess.close()
