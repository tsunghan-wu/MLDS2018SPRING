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

def model1(x):
	'''
	conv (5 14*14filter)	28*28 -> 28*28 *5 (padding)
	pool (2*2)	28*28 *32 -> 14*14 *5

	Flatten()

	dense1  14*14*5 -> 50
	output  50 -> 10

	params =  CNN : 5*14*14 + 5
	params =  DNN : 5*14*14 * 50 + 50*10 + 50
	'''

	x_image = tf.reshape(x, [-1, 28, 28, 1])

	### conv layer 1 ####
	w_conv1 = init_weights([14,14,1,5])
	b_conv1 = init_weights([5])
	
	#### pool layer 1 ####
	h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	#### Flatten layer ####
	h_flat = tf.reshape(h_pool1, [ -1 , 14 * 14 * 5 ])

	#### dense1 ####
	w_fc1 = init_weights([ 14 * 14 * 5 , 50])
	b_fc1 = init_weights([50])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat , w_fc1) + b_fc1)

	#### dropout layer ####
	#h_fc1_drop = tf.nn.dropout(h_fc1,0.5)

	#### output ####
	W_fc2 = init_weights([50, 10])
	b_fc2 = init_weights([10])
	y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
	
	return y_conv

def model2(x):
	'''
	conv (5 14*14filter)	28*28 -> 28*28 *5 (padding)
	pool (2*2)	28*28 *32 -> 14*14 *5

	Flatten()

	dense1  14*14*5 -> 48
	dense2  48 -> 42
	output  42 -> 10

	params CNN : 5*14*14 = 980 + 5
	params DNN : 5*14*14 * 48 + 48*42 + 42*10 + 48 + 42
	'''

	x_image = tf.reshape(x, [-1, 28, 28, 1])

	### conv layer 1 ####
	w_conv1 = init_weights([14,14,1,5])
	b_conv1 = init_weights([5])
	
	#### pool layer 1 ####
	h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	#### Flatten layer ####
	h_flat = tf.reshape(h_pool1, [ -1 , 14 * 14 * 5 ])

	#### dense1 ####
	w_fc1 = init_weights([ 14 * 14 * 5 , 48])
	b_fc1 = init_weights([48])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat , w_fc1) + b_fc1)

	#### output ####
	W_fc2 = init_weights([48, 42])
	b_fc2 = init_weights([42])
	h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
	
	W_fc3 = init_weights([42, 10])
	b_fc3 = init_weights([10])
	y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3
	
	return y_conv


def model3(x):
	'''
	conv (32 5*5filter)	28*28 -> 28*28 *32 (padding)
	pool (2*2)	28*28 *32 -> 14*14 *32
	conv (20 3*3filter)	14*14 *32 -> 14*14 *20 (padding)
	pool (2*2)	14*14 *20 -> 7*7*20

	Flatten()

	dense1  7*7*20 -> 48
	dense2  48 -> 42
	output  42 -> 10

	params CNN : 32*5*5 + 20*3*3 = 980  + 52
	params DNN : 20*7*7 * 48 + 48*42 + 42*10 + 48 + 42
	'''

	x_image = tf.reshape(x, [-1, 28, 28, 1])

	### conv layer 1 ####
	w_conv1 = init_weights([5,5,1,32])
	b_conv1 = init_weights([32])
	
	#### pool layer 1 ####
	h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	### conv layer 2 ####
	w_conv2 = init_weights([3,3,32,20])
	b_conv2 = init_weights([20])
	h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
	#### pool layer 2 ####
	h_pool2 = max_pool_2x2(h_conv2)
	


	#### Flatten layer ####
	h_flat = tf.reshape(h_pool2, [ -1 , 7 * 7 * 20 ])

	#### dense1 ####
	w_fc1 = init_weights([ 7 * 7 * 20 , 48])
	b_fc1 = init_weights([48])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat , w_fc1) + b_fc1)

	#### output ####
	W_fc2 = init_weights([48, 42])
	b_fc2 = init_weights([42])
	h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
	
	W_fc3 = init_weights([42, 10])
	b_fc3 = init_weights([10])
	y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3
	
	return y_conv
