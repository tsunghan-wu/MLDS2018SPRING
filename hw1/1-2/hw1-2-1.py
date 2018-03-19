import numpy as np
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from util import seq


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# set training data (x, y)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

model = seq(x , y_ , 784)
for _ in range(2):
	model.add_FC(10)
	model.add_activate(tf.nn.relu)

model.add_FC(10)
sess = tf.Session()
pred_y_ , train_step = model.get_train(sess)
tf.global_variables_initializer().run(session=sess)
print("\n{0:-^40s}\n".format("all param:" + str(model.summary())))
Acc = np.empty(shape=[0, 1])
for _ in range(2000):
	trainX , trainY = mnist.train.next_batch(1000)
	sess.run(train_step,feed_dict={
			x : trainX,
			y_ : trainY
		})
	if _ % 3 == 0:
		model.save_whole_variable("csvdir/whole_1.csv")
		model.save_one_layer("csvdir/L1_1.csv")
		accuracy = model.get_acc(mnist.train.images, mnist.train.labels)
		Acc = np.append(Acc, np.array([accuracy])).reshape(-1, 1)
	if _ % 100 == 0 :
		accuracy = model.get_acc(mnist.train.images, mnist.train.labels)
		print ("epoch %d acc %8g " %(_,accuracy))

np.savetxt("csvdir/acc_1.csv", Acc)
