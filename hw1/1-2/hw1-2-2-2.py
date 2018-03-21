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
for _ in range(3):
	model.add_FC(10)
	model.add_activate(tf.nn.relu)

###########################
seed = 31416
tf.set_random_seed(seed)
###########################
model.add_FC(1)
sess = tf.Session()
pred_y_ , train_step = model.get_train(sess)
tf.global_variables_initializer().run(session=sess)
print("\n{0:-^40s}\n".format("all param:" + str(model.summary())))

Loss = np.empty(shape=[0, 1])
grad_norm = np.empty(shape=[0, 1])


def gradient(sess, trainX, trainY):
	grad_all = 0.0
	for p in tf.trainable_variables():
		grad = sess.run(tf.gradients(xs=p, ys=model.loss), feed_dict={x:trainX, y_:trainY})
		grad = np.sum(np.square(grad))
		grad_all += grad
	return (grad_all ** 0.5)

for _ in range(4000):
	trainX , trainY = next_batch(10000,range=def_range)
	sess.run(train_step,feed_dict={
			x : trainX,
			y_ : trainY
		})
	norm = gradient(sess, trainX, trainY)
	loss = model.get_loss(allX.reshape(-1,1), allY.reshape(-1,1))
	Loss = np.append(Loss, np.array([loss])).reshape(-1, 1)
	grad_norm = np.append(grad_norm, np.array([norm])).reshape(-1, 1)
	if _ % 100 == 0 :
		print ("epoch %d acc %8g " %(_,loss))

# np.savetxt("csvdir/Floss_" + str(times) + ".csv", Loss)
np.savetxt("csvdir/1-2-2_acc.csv", Acc)
np.savetxt("csvdir/1-2-2_grad_norm.csv", grad_norm)