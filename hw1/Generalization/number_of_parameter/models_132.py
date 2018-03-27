import tensorflow as tf 
import models
def init_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def conv2d(x, W):
	return tf.nn.conv2d(
		x, W, strides=[1, 1, 1, 1], padding = 'SAME')
	# 1 [x_strides , y_strides] 1
	# padding:  output size + padding = input_size 

def max_pool_2x2(x):
	return tf.nn.max_pool(
		x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	# 1 [x_strides , y_strides] 1

def model(x,params=[700,700,700,10]):
	x_image = tf.reshape(x, [-1, 28 * 28])

	neuron = params
	inc , inn = x_image , 28*28
	for out in neuron:
		w = init_weights([inn, out])
		b = init_weights([out])
		inc = tf.matmul(inc , w) + b
		if inn == 28*28:
			inc = tf.nn.relu(inc)
		inn = out
	return inc
