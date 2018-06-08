import tensorflow as tf 
import numpy as np

def conv2d(name, tensor,ksize, out_dim, stddev=0.01, stride=2, padding='SAME'):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [ksize, ksize, tensor.get_shape()[-1],out_dim], dtype=tf.float32,
							initializer=tf.random_normal_initializer(stddev=stddev))
		var = tf.nn.conv2d(tensor,w,[1,stride, stride,1],padding=padding)
		b = tf.get_variable('b', [out_dim], 'float32',initializer=tf.constant_initializer(0.01))
		return tf.nn.bias_add(var, b)

def deconv2d(name, tensor, ksize, outshape, stddev=0.01, stride=2, padding='SAME'):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [ksize, ksize, outshape[-1], tensor.get_shape()[-1]], dtype=tf.float32,
							initializer=tf.random_normal_initializer(stddev=stddev))
		var = tf.nn.conv2d_transpose(tensor, w, outshape, strides=[1, stride, stride, 1], padding=padding)
		b = tf.get_variable('b', [outshape[-1]], 'float32', initializer=tf.constant_initializer(0.01))
		return tf.nn.bias_add(var, b)

def fully_connected(name,value, output_shape):
	with tf.variable_scope(name, reuse=None) as scope:
		shape = value.get_shape().as_list()
		w = tf.get_variable('w', [shape[1], output_shape], dtype=tf.float32,
									initializer=tf.random_normal_initializer(stddev=0.01))
		b = tf.get_variable('b', [output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

		return tf.matmul(value, w) + b


def batch_norm(input, is_training, momentum=0.9, epsilon=2e-5, in_place_update=True, name="batch_norm"):
	if in_place_update:
		return tf.contrib.layers.batch_norm(input,
										decay=momentum,
										center=True,
										scale=True,
										epsilon=epsilon,
										updates_collections=None,
										is_training=is_training,
										scope=name)
	else:
		return tf.contrib.layers.batch_norm(input,
										decay=momentum,
										center=True,
										scale=True,
										epsilon=epsilon,
										is_training=is_training,
										scope=name)