import numpy as np
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from util import seq


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# set training data (x, y)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
seed = 3141644
tf.set_random_seed(seed)

model = seq(x , y_ , 784)
for _ in range(2):
	model.add_FC(10)
	model.add_activate(tf.nn.relu)

model.add_FC(10)

sess = tf.Session()
pred_y_ , train_step = model.get_train(sess,tf.train.AdamOptimizer(1e-3))
tf.global_variables_initializer().run(session=sess)
print("\n{0:-^40s}\n".format("all param:" + str(model.summary())))


Acc = []
Loss = []
grad_norm = []
# var_grad = tf.gradients(loss, x)[0]
for _ in range(30000):
	trainX , trainY = mnist.train.next_batch(100)
	sess.run(train_step,feed_dict={
			x : trainX,
			y_ : trainY
		})
	norm = model.get_grad_norm(trainX,trainY)
	accuracy = model.get_acc(mnist.train.images, mnist.train.labels)
	loss = model.get_loss(mnist.train.images, mnist.train.labels)
	###
	Acc.append(accuracy)
	Loss.append(loss)
	grad_norm.append(norm)
	if _ % 10 == 0:
		print ("epoch %d acc %8g  , norm %g " %(_,accuracy , norm))

np.savetxt("csvdir/1-2_acc.csv", np.array(Acc).reshape(-1,1))
np.savetxt("csvdir/1-2_grad_norm.csv", np.array(grad_norm).reshape(-1,1))
np.savetxt("csvdir/1-2_loss.csv", np.array(Loss).reshape(-1,1))
