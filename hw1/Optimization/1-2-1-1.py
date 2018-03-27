import numpy as np
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from util import seq

########### Function Preparing ############
f_a = 0.5
f_b = 2
def function(x):
	'''
	#function 1
	y = x * 0
	for n in range(10):
		y += np.power(f_a ,n) * np.cos( np.power(f_b,n) * np.pi * x)
	return y
	'''
	#funciton2
	return np.sin(x) + np.cos(x*x)

def_range = (0,10)
print('{0:-^40s}'.format("first generate"))
allX = np.arange(def_range[0],def_range[1],0.001)
allY = function(allX)
	
def next_batch(size,range=(0,1)):
	b = np.random.randint(0,len(allX),size)
	return allX[b].reshape(-1,1) , allY[b].reshape(-1,1)


# set training data (x, y)
x = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [None, 1])

model = seq(x , y_ , 1)
for _ in range(4):
	model.add_FC(10)
	model.add_activate(tf.nn.relu)

###########################
times = 1
###########################
model.add_FC(1)
sess = tf.Session()
pred_y_ , train_step = model.get_train(sess)
tf.global_variables_initializer().run(session=sess)
print("\n{0:-^40s}\n".format("all param:" + str(model.summary())))
exit()
Loss = np.empty(shape=[0, 1])
for _ in range(7000):
	trainX , trainY = next_batch(10000,range=def_range)
	sess.run(train_step,feed_dict={
			x : trainX,
			y_ : trainY
		})
	if _ % 3 == 0:
		loss = model.get_loss(allX.reshape(-1,1), allY.reshape(-1,1))
		model.save_whole_variable("csvdir/Fwhole_" + str(times) + ".csv")
		model.save_one_layer("csvdir/FL1_" + str(times) + ".csv")
		# accuracy = model.get_acc(mnist.train.images, mnist.train.labels)
		Loss = np.append(Loss, np.array([loss])).reshape(-1, 1)
	if _ % 100 == 0:
		print ("epoch = %d, loss = %g"%(_,model.get_loss(allX.reshape(-1,1), allY.reshape(-1,1))))

np.savetxt("csvdir/Floss_" + str(times) + ".csv", Loss)
