import tensorflow as tf 
import models
def init_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def model(x):
	x_image = tf.reshape(x, [-1, 28 * 28])

	neuron = [700,700,700]
	#### dense1 ####
	w_fc1 = init_weights([ 28*28, neuron[0]])
	b_fc1 = init_weights([neuron[0]])
	h_fc1 = tf.nn.relu(tf.matmul(x_image , w_fc1) + b_fc1)

	#### output ####
	w_fc2 = init_weights([neuron[0], neuron[1]])
	b_fc2 = init_weights([neuron[1]])
	h_fc2 = tf.matmul(h_fc1, w_fc2) + b_fc2
	
	w_fc3 = init_weights([neuron[1], neuron[2]])
	b_fc3 = init_weights([neuron[2]])
	h_fc3 = tf.matmul(h_fc2, w_fc3) + b_fc3
	
	w_fc4 = init_weights([neuron[2],10])
	b_fc4 = init_weights([10])
	y_conv = tf.matmul(h_fc3, w_fc4) + b_fc4
	
	return y_conv
