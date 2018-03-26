import numpy as np
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
<<<<<<< HEAD
=======
from util import seq

>>>>>>> 910b511e78ff7ede38926da85ab887eb42402a3b

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# set training data (x, y)
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

model = seq(x , y_ , 784)
for _ in range(2):
	model.add_FC(10)
	model.add_activate(tf.nn.relu)

###########################
times = 6
seed = 2718
<<<<<<< HEAD
times = 7
seed = 7122
=======
>>>>>>> 910b511e78ff7ede38926da85ab887eb42402a3b
###########################
tf.set_random_seed(seed)
model.add_FC(10)
sess = tf.Session()
pred_y_ , train_step = model.get_train(sess)
tf.global_variables_initializer().run(session=sess)
print("\n{0:-^40s}\n".format("all param:" + str(model.summary())))
Acc = np.empty(shape=[0, 1])
<<<<<<< HEAD
output = []
=======
>>>>>>> 910b511e78ff7ede38926da85ab887eb42402a3b
for _ in range(4000):
	trainX , trainY = mnist.train.next_batch(1000)
	sess.run(train_step,feed_dict={
			x : trainX,
			y_ : trainY
		})
	if _ % 3 == 0:
<<<<<<< HEAD
		#model.save_whole_variable("csvdir/whole_" + str(times) + ".csv")
		#model.save_one_layer("csvdir/L1_" + str(times) + ".csv")
		accuracy = model.get_acc(mnist.train.images, mnist.train.labels)
		Acc = np.append(Acc, np.array([accuracy])).reshape(-1, 1)
		######
		output.append(model.get_whole_variable().reshape(-1))
=======
		model.save_whole_variable("csvdir/whole_" + str(times) + ".csv")
		model.save_one_layer("csvdir/L1_" + str(times) + ".csv")
		accuracy = model.get_acc(mnist.train.images, mnist.train.labels)
		Acc = np.append(Acc, np.array([accuracy])).reshape(-1, 1)
>>>>>>> 910b511e78ff7ede38926da85ab887eb42402a3b
	if _ % 100 == 0 :
		accuracy = model.get_acc(mnist.train.images, mnist.train.labels)
		print ("epoch %d acc %8g " %(_,accuracy))

np.savetxt("csvdir/acc_" + str(times) + ".csv", Acc)
<<<<<<< HEAD
###
output = np.array(output)
Acc = np.array(Acc)
np.save('output_par' + str(times),np.array(output))
np.save('output_acc' + str(times),np.array(Acc))
=======
>>>>>>> 910b511e78ff7ede38926da85ab887eb42402a3b
