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
	conv (120 3*3filter)	32 * 32 * 3  -> 32 *32 * 120 (padding)
	pool (2*2)	32 * 32 *120 ->  16 * 16 * 120

	conv (8 3*3filter)	16 * 16 * 120  -> 16 *16 * 8 (padding)
	pool (2*2)	16 * 16 * 8 ->  8 * 8 * 8
	
	Flatten()

	dense1  8 * 8 * 8  -> 40

	output  40 -> 10
	'''

	x_image = tf.reshape(x, [-1, 32, 32, 3 ] )

	### conv layer 1 ####
	w_conv1 = init_weights([ 3 , 3 , 3 , 120])
	b_conv1 = init_weights([ 120 ])
	
	#### pool layer 1 ####
	h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	### conv layer 2 ####
	w_conv2 = init_weights([ 3 , 3 , 120 , 8])
	b_conv2 = init_weights([ 8 ])

	#### pool layer 2 ####
	h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	#### Flatten layer ####
	h_flat = tf.reshape(h_pool2, [ -1 , 8 * 8 * 8 ])


	#### dense1 ####
	w_fc1 = init_weights([ 8 * 8 * 8 , 40])
	b_fc1 = init_weights([40])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat , w_fc1) + b_fc1)

	#### output ####
	W_fc2 = init_weights([40, 10])
	b_fc2 = init_weights([10])
	y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
	
	return y_conv


def model2(x):
	'''
	conv (120 3*3filter)	32 * 32 * 3  -> 32 *32 * 120 (padding)
	pool (2*2)	32 * 32 *120 ->  16 * 16 * 120

	conv (8 3*3filter)	16 * 16 * 120  -> 16 *16 * 8 (padding)
	pool (2*2)	16 * 16 * 8 ->  8 * 8 * 8

	Flatten()

	dense1  8 * 8 * 8  -> 32
	dense2  32 -> 64		
	dense3  64 -> 32

	output  32 -> 10
	'''

	x_image = tf.reshape(x, [-1, 32, 32, 3 ] )

	### conv layer 1 ####
	w_conv1 = init_weights([ 3 , 3 , 3 , 120])
	b_conv1 = init_weights([ 120 ])
	
	#### pool layer 1 ####
	h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	### conv layer 2 ####
	w_conv2 = init_weights([ 3 , 3 , 120 , 8])
	b_conv2 = init_weights([ 8 ])

	#### pool layer 2 ####
	h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	#### Flatten layer ####
	h_flat = tf.reshape(h_pool2, [ -1 , 8 * 8 * 8 ])

	#### dense1 ####
	w_fc1 = init_weights([ 8 * 8 * 8 , 32])
	b_fc1 = init_weights([32])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat , w_fc1) + b_fc1)

	#### dense2 ####
	w_fc2 = init_weights([ 32 , 64])
	b_fc2 = init_weights([64])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1 , w_fc2) + b_fc2)


	#### dense3 ####
	w_fc3 = init_weights([ 64 , 32])
	b_fc3 = init_weights([32])
	h_fc3 = tf.nn.relu(tf.matmul(h_fc2 , w_fc3) + b_fc3)

	#### output ####
	w_fc4 = init_weights([32, 10])
	b_fc4 = init_weights([10])
	y_conv = tf.matmul(h_fc3, w_fc4) + b_fc4

	return y_conv

def model3(x):
	'''
	conv (32 3*3filter)	32 * 32 * 3  -> 32 *32 * 32 (padding)
	pool (2*2)	32 * 32 * 32 ->  16 * 16 * 32

	conv (64 3*3filter)	16 * 16 * 32  -> 16 *16 * 64 (padding)
	pool (2*2)	16 * 16 * 64 ->  8 * 8 * 64

	conv (32 3*3filter) 8 * 8 * 64 -> 8 * 8 * 32  (padding)
	pool (2*2)  8 * 8 * 32 -> 4 * 4 * 32


	Flatten()

	dense1  4 * 4 * 32  -> 32
	dense2  32 -> 64		
	dense3  64 -> 32

	output  32 -> 10
	'''

	x_image = tf.reshape(x, [-1, 32, 32, 3 ] )

	### conv layer 1 ####
	w_conv1 = init_weights([ 3 , 3 , 3 , 32])
	b_conv1 = init_weights([ 32 ])
	
	#### pool layer 1 ####
	h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	### conv layer 2 ####
	w_conv2 = init_weights([ 3 , 3 , 32 , 64])
	b_conv2 = init_weights([ 64 ])

	#### pool layer 2 ####
	h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)


	### conv layer 2 ####
	w_conv3 = init_weights([ 3 , 3 , 64 , 32])
	b_conv3 = init_weights([ 32 ])

	#### pool layer 2 ####
	h_conv3 = tf.nn.relu(conv2d(h_pool2,w_conv3) + b_conv3)
	h_pool3 = max_pool_2x2(h_conv3)


	#### Flatten layer ####
	h_flat = tf.reshape(h_pool3, [ -1 , 4 * 4 * 32 ])

	#### dense1 ####
	w_fc1 = init_weights([ 4 * 4 * 32 , 32])
	b_fc1 = init_weights([32])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat , w_fc1) + b_fc1)

	#### dense2 ####
	w_fc2 = init_weights([ 32 , 64])
	b_fc2 = init_weights([64])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1 , w_fc2) + b_fc2)


	#### dense3 ####
	w_fc3 = init_weights([ 64 , 32])
	b_fc3 = init_weights([32])
	h_fc3 = tf.nn.relu(tf.matmul(h_fc2 , w_fc3) + b_fc3)

	#### output ####
	w_fc4 = init_weights([32, 10])
	b_fc4 = init_weights([10])
	y_conv = tf.matmul(h_fc3, w_fc4) + b_fc4

	return y_conv

def model4(x):
	'''
	conv (32 3*3filter)	32 * 32 * 3  -> 32 *32 * 32 (padding)
	pool (2*2)	32 * 32 * 32 ->  16 * 16 * 32

	conv (64 3*3filter)	16 * 16 * 32  -> 16 *16 * 64 (padding)
	pool (2*2)	16 * 16 * 64 ->  8 * 8 * 64

	conv (32 3*3filter) 8 * 8 * 64 -> 8 * 8 * 32  (padding)
	pool (2*2)  8 * 8 * 32 -> 4 * 4 * 32


	Flatten()

	dense1  4 * 4 * 32  -> 32
	dense2  32 -> 64		
	dense3  64 -> 32

	output  32 -> 10
	'''

	x_image = tf.reshape(x, [-1, 32, 32, 3 ] )

	### conv layer 1 ####
	w_conv1 = init_weights([ 3 , 3 , 3 , 32])
	b_conv1 = init_weights([ 32 ])
	
	#### pool layer 1 ####
	h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	### conv layer 2 ####
	w_conv2 = init_weights([ 3 , 3 , 32 , 32])
	b_conv2 = init_weights([ 32 ])

	#### pool layer 2 ####
	h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)


	### conv layer 3 ####
	w_conv3 = init_weights([ 3 , 3 , 32 , 28])
	b_conv3 = init_weights([ 28 ])

	#### pool layer 3 ####
	h_conv3 = tf.nn.relu(conv2d(h_pool2,w_conv3) + b_conv3)
	h_pool3 = max_pool_2x2(h_conv3)


	#### Flatten layer ####
	h_flat = tf.reshape(h_pool3, [ -1 , 4 * 4 * 28 ])

	#### dense1 ####
	w_fc1 = init_weights([ 4 * 4 * 28 , 30])
	b_fc1 = init_weights([30])
	h_fc1 = tf.nn.relu(tf.matmul(h_flat , w_fc1) + b_fc1)

	#### dense2 ####
	w_fc2 = init_weights([ 30 , 30])
	b_fc2 = init_weights([30])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1 , w_fc2) + b_fc2)


	#### output ####
	w_fc3 = init_weights([ 30 , 10])
	b_fc3 = init_weights([10])
	y_conv = tf.matmul(h_fc2 , w_fc3) + b_fc3


	return y_conv