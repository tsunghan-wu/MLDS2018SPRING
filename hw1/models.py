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
	conv (32 5*5filter)	28*28 -> 28*28 *32 (padding)
	pool (2*2)	28*28 *32 -> 14*14 *32

	conv (64 5*5filter) 14*14 *32-> 14*14 *64 (padding)
	pool (2*2) 14*14 *64 -> 7*7 *64

	Flatten()

	dense1  7*7*64 -> 256
	output  256 -> 10
	'''
	d1 = 32

	x_image = tf.reshape(x, [-1, 28, 28, 1])

	### conv layer 1 ####
	w_conv1 = init_weights([5,5,1,32])
	b_conv1 = init_weights([32])
	
	#### pool layer 1 ####
	h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	### conv layer 2 ####
	w_conv2 = init_weights([5,5,32,64])
	b_conv2 = init_weights([64])
	# 5*5 patch , input channel 32 , output channel 64

	#### pool layer 2 ####
	h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)


	#### Flatten layer ####
	h_flat = tf.reshape(h_pool2, [ -1 , 7 * 7 * 64 ])

	#### dense1 ####
	w_fc1 = init_weights([ 7 * 7 * 64 , d1])
	b_fc1 = init_weights([d1])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat , w_fc1) + b_fc1)

	#### dropout layer ####
	#h_fc1_drop = tf.nn.dropout(h_fc1,0.5)

	#### output ####
	W_fc2 = init_weights([d1, 10])
	b_fc2 = init_weights([10])
	y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
	
	return y_conv


def model2(x):
	'''
	conv (32 5*5filter)	28*28 -> 28*28 *32 (padding)
	pool (2*2)	28*28 *32 -> 14*14 *32

	conv (64 5*5filter) 14*14 *32-> 14*14 *64 (padding)
	pool (2*2) 14*14 *64 -> 7*7 *64

	Flatten()

	dense1  7*7*64 -> 128
	dense2  128 -> 128
	dense3  128 -> 128
	output  128 -> 10
	'''
	d =  16
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	### conv layer 1 ####
	w_conv1 = init_weights([5,5,1,32])
	b_conv1 = init_weights([32])
	
	#### pool layer 1 ####
	h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	### conv layer 2 ####
	w_conv2 = init_weights([5,5,32,64])
	b_conv2 = init_weights([64])
	# 5*5 patch , input channel 32 , output channel 64

	#### pool layer 2 ####
	h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)


	#### Flatten layer ####
	h_flat = tf.reshape(h_pool2, [ -1 , 7 * 7 * 64 ])

	#### dense1 ####
	w_fc1 = init_weights([ 7 * 7 * 64 , d])
	b_fc1 = init_weights([d])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat , w_fc1) + b_fc1)

	#### dense2 ####
	w_fc2 = init_weights([d  , d])
	b_fc2 = init_weights([d])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1 , w_fc2) + b_fc2)

	#### dense3 ####
	w_fc3 = init_weights([d  , d])
	b_fc3 = init_weights([d])
	h_fc3 = tf.nn.relu(tf.matmul(h_fc2 , w_fc3) + b_fc3)

	#### dropout layer ####
	#h_fc1_drop = tf.nn.dropout(h_fc1,0.5)

	#### output ####
	W_fc2 = init_weights([d, 10])
	b_fc2 = init_weights([10])
	y_conv = tf.matmul(h_fc3, W_fc2) + b_fc2
	
	return y_conv



def model3(x):
	'''
	conv (32 5*5filter)	28*28 -> 28*28 *32 (padding)
	conv (32 5*5filter) 28*28*32-> 28*28 *32 (padding)
	conv (32 5*5filter)	28*28*32 -> 28*28 *32 (padding)
	conv (32 5*5filter) 28*28*32-> 28*28 *32 (padding)
	
	Flatten()

	dense1  7*7*64 -> 128
	dense2  128 -> 128
	dense3  128 -> 128
	output  128 -> 10
	'''
	d = 16
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	### conv layer 1 ####
	w_conv1 = init_weights([5,5,1,32])
	b_conv1 = init_weights([32])
	h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
	
	### conv layer 2 ####
	w_conv2 = init_weights([5,5,32,32])
	b_conv2 = init_weights([32])
	h_conv2 = tf.nn.relu(conv2d(h_conv1,w_conv2) + b_conv2)
	
	### conv layer 3 ####
	w_conv3 = init_weights([5,5,32,32])
	b_conv3 = init_weights([32])
	h_conv3 = tf.nn.relu(conv2d(h_conv2,w_conv3) + b_conv3)
	
	### conv layer 4 ####
	w_conv4 = init_weights([5,5,32,32])
	b_conv4 = init_weights([32])
	h_conv4 = tf.nn.relu(conv2d(h_conv3,w_conv4) + b_conv4)
	

	#### Flatten layer ####
	h_flat = tf.reshape(h_conv4, [ -1 , 28 * 28 * 32 ])

	#### dense1 ####
	w_fc1 = init_weights([ 28 * 28 * 32 , d])
	b_fc1 = init_weights([d])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat , w_fc1) + b_fc1)

	#### dense2 ####
	w_fc2 = init_weights([d  , d])
	b_fc2 = init_weights([d])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1 , w_fc2) + b_fc2)

	#### dense3 ####
	w_fc3 = init_weights([d  , d])
	b_fc3 = init_weights([d])
	h_fc3 = tf.nn.relu(tf.matmul(h_fc2 , w_fc3) + b_fc3)

	#### dropout layer ####
	#h_fc1_drop = tf.nn.dropout(h_fc1,0.5)

	#### output ####
	W_fc2 = init_weights([d, 10])
	b_fc2 = init_weights([10])
	y_conv = tf.matmul(h_fc3, W_fc2) + b_fc2
	
	return y_conv


def model4(x):
	'''

	conv (32 5*5filter)	28*28 -> 28*28 *32 (padding)
	conv (32 5*5filter) 28*28*32-> 28*28 *32 (padding)
	conv (32 5*5filter)	28*28*32 -> 28*28 *32 (padding)
	conv (32 5*5filter) 28*28*32-> 28*28 *32 (padding)
	
	Flatten()

	dense1  7*7*64 -> 128
	dense2  128 -> 128
	dense3  128 -> 128
	output  128 -> 10
	

	Flatten()

	dense1  7*7*64 -> 128
	dense2  128 -> 128
	dense3  128 -> 128
	output  128 -> 10
	'''
	d = 16
	x_image = tf.reshape(x, [-1, 28, 28, 1])

	### conv layer 1 ####
	w_conv1 = init_weights([5,5,1,32])
	b_conv1 = init_weights([32])
	h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
	
	### conv layer 2 ####
	w_conv2 = init_weights([5,5,32,32])
	b_conv2 = init_weights([32])
	h_conv2 = tf.nn.relu(conv2d(h_conv1,w_conv2) + b_conv2)
	
	### conv layer 3 ####
	w_conv3 = init_weights([5,5,32,32])
	b_conv3 = init_weights([32])
	h_conv3 = tf.nn.relu(conv2d(h_conv2,w_conv3) + b_conv3)
	
	h_residual = h_conv1 + h_conv3
	### conv layer 4 ####
	w_conv4 = init_weights([5,5,32,32])
	b_conv4 = init_weights([32])
	h_conv4 = tf.nn.relu(conv2d(h_residual,w_conv4) + b_conv4)
	

	#### Flatten layer ####
	h_flat = tf.reshape(h_conv4, [ -1 , 28 * 28 * 32 ])

	#### dense1 ####
	w_fc1 = init_weights([ 28 * 28 * 32 , d])
	b_fc1 = init_weights([d])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat , w_fc1) + b_fc1)

	#### dense2 ####
	w_fc2 = init_weights([d  , d])
	b_fc2 = init_weights([d])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1 , w_fc2) + b_fc2)

	#### dense3 ####
	w_fc3 = init_weights([d  , d])
	b_fc3 = init_weights([d])
	h_fc3 = tf.nn.relu(tf.matmul(h_fc2 , w_fc3) + b_fc3)

	#### dropout layer ####
	#h_fc1_drop = tf.nn.dropout(h_fc1,0.5)

	#### output ####
	W_fc2 = init_weights([d, 10])
	b_fc2 = init_weights([10])
	y_conv = tf.matmul(h_fc3, W_fc2) + b_fc2
	
	return y_conv