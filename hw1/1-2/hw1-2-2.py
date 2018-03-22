import numpy as np
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from util import seq


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# set training data (x, y)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
seed = 31416
tf.set_random_seed(seed)
model = seq(x , y_ , 784)
for _ in range(2):
	model.add_FC(10)
	model.add_activate(tf.nn.relu)

model.add_FC(10)
sess = tf.Session()
pred_y_ , train_step = model.get_train(sess)
tf.global_variables_initializer().run(session=sess)
print("\n{0:-^40s}\n".format("all param:" + str(model.summary())))

def gradient(sess, trainX, trainY):
	grad_all = []
	grad = sess.run(model.grad_norm, feed_dict={x:trainX, y_:trainY})
	for lay in grad:
		grad_all += lay.flatten().tolist()
	grad_all = np.sqrt(np.sum(np.square(grad_all)))
	return grad_all 


Acc = np.empty(shape=[0, 1])
grad_norm = np.empty(shape=[0, 1])
# var_grad = tf.gradients(loss, x)[0]
for _ in range(10000):
	trainX , trainY = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={
			x : trainX,
			y_ : trainY
		})
	norm = gradient(sess, trainX, trainY)
	accuracy = model.get_acc(mnist.train.images, mnist.train.labels)
	Acc = np.append(Acc, np.array([accuracy])).reshape(-1, 1)
	grad_norm = np.append(grad_norm, np.array([norm])).reshape(-1, 1)
	if _ % 100 == 0:
		print (norm)
		print ("epoch %d acc %8g " %(_,accuracy))

np.savetxt("csvdir/1-2_acc.csv", Acc)
np.savetxt("csvdir/1-2_grad_norm.csv", grad_norm)
