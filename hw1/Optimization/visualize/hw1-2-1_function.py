import numpy as np
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from util import seq

def function(x):
	return np.sin(x) + np.cos(x**2)

def next_batch(size,range=(0,1)):
	bx = np.random.rand(size)*(range[1]-range[0]) +range[0]
	by = function(bx)
	return bx.reshape(-1,1),by.reshape(-1,1)

# set training data (x, y)
x = tf.placeholder(tf.float32, [None, 1])
y_ = tf.placeholder(tf.float32, [None, 1])

model = seq(x , y_ , 1)
for _ in range(4):
	model.add_FC(10)
	model.add_activate(tf.nn.relu)

###########################
times = sys.argv[1]
###########################
model.add_FC(1)
sess = tf.Session()
pred_y_ , train_step = model.get_train(sess)
tf.global_variables_initializer().run(session=sess)
print("\n{0:-^40s}\n".format("all param:" + str(model.summary())))
Loss = np.empty(shape=[0, 1])
for _ in range(5000):
	trainX , trainY = next_batch(100,range=(0,10))
	sess.run(train_step,feed_dict={
			x : trainX,
			y_ : trainY
		})
	if _ % 3 == 0:
		model.save_whole_variable("csvdir/2whole_" + str(times) + ".csv")
		model.save_one_layer("csvdir/2L1_" + str(times) + ".csv")
		loss = model.get_loss(trainX, trainY)
		Loss = np.append(Loss, np.array([loss])).reshape(-1, 1)
	if _ % 100 == 0 :
		loss = model.get_loss(trainX, trainY)
		print ("epoch %d loss %8g " %(_,loss))

np.savetxt("csvdir/2loss_" + str(times) + ".csv", Loss)
