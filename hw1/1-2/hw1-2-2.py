import numpy as np
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from util import seq


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# set training data (x, y)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
<<<<<<< HEAD
seed = 3141644
tf.set_random_seed(seed)

model = seq(x , y_ , 784)
for _ in range(2):
	model.add_FC(10,stddev=1)
	model.add_activate(tf.nn.relu)

model.add_FC(10,stddev=1)

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
=======
seed = 31416
tf.set_random_seed(seed)
model = seq(x , y_ , 784)
for _ in range(3):
	model.add_FC(100)
	model.add_activate(tf.nn.relu)

model.add_FC(10)
sess = tf.Session()
pred_y_ , train_step = model.get_train(sess)
tf.global_variables_initializer().run(session=sess)
print("\n{0:-^40s}\n".format("all param:" + str(model.summary())))

def gradient(sess, trainX, trainY):
	grad_all = 0.0
	for p in tf.trainable_variables():
		grad = sess.run(tf.gradients(xs=p, ys=model.loss), feed_dict={x:trainX, y_:trainY})
		grad = np.sum(np.square(grad))
		grad_all += grad
	return (grad_all ** 0.5)


Acc = np.empty(shape=[0, 1])
grad_norm = np.empty(shape=[0, 1])
# var_grad = tf.gradients(loss, x)[0]
for _ in range(10000):
	trainX , trainY = mnist.train.next_batch(1000)
>>>>>>> 910b511e78ff7ede38926da85ab887eb42402a3b
	sess.run(train_step,feed_dict={
			x : trainX,
			y_ : trainY
		})
<<<<<<< HEAD
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
=======
	norm = gradient(sess, trainX, trainY)
	accuracy = model.get_acc(mnist.train.images, mnist.train.labels)
	Acc = np.append(Acc, np.array([accuracy])).reshape(-1, 1)
	grad_norm = np.append(grad_norm, np.array([norm])).reshape(-1, 1)
	if _ % 100 == 0:
		print (norm)
		print ("epoch %d acc %8g " %(_,accuracy))

np.savetxt("csvdir/1-2_acc.csv", Acc)
np.savetxt("csvdir/1-2_grad_norm.csv", grad_norm)
>>>>>>> 910b511e78ff7ede38926da85ab887eb42402a3b
