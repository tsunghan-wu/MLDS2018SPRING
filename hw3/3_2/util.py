import tensorflow as tf 
import numpy as np
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,	name="conv2d" , padding='SAME'):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
			initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)
		b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(0.0))
		conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
		return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d"):
	with tf.variable_scope(name):
		# filter : [height, width, output_channels, in_channels]
		w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],initializer=tf.random_normal_initializer(stddev=stddev))
		deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,strides=[1, d_h, d_w, 1])
		b = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
		deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

		return deconv
'''
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
'''
def tag_deconv2d(name, tensor, ksize, inshape,outshape, tag,stddev=0.01, stride=2, padding='SAME'):
	with tf.variable_scope(name):
		
		tag = tf.reshape(tag,[1,-1])
		pre_w = tf.get_variable('w', [tag.get_shape()[-1] , ksize* ksize*outshape[-1]*inshape], 
			dtype=tf.float32,initializer=tf.random_normal_initializer(stddev=stddev))
		w = tf.matmul(tag,pre_w)
		
		w = tf.reshape(w,[ksize, ksize, outshape[-1], tensor.get_shape()[-1]])
		var = tf.nn.conv2d_transpose(tensor, w, outshape, strides=[1, stride, stride, 1], padding=padding)
		b = tf.get_variable('b', [outshape[-1]], 'float32', initializer=tf.constant_initializer(0.01))
		return tf.nn.bias_add(var, b)

def conv3d(name, tensor,ksize, hsize,out_dim, stddev=0.01, stride=2,h_stride=1 ,padding='SAME'):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [ksize, ksize, hsize,tensor.get_shape()[-1],out_dim], dtype=tf.float32,
							initializer=tf.random_normal_initializer(stddev=stddev))
		var = tf.nn.conv3d(tensor,w,[1,stride, stride, h_stride,1],padding=padding)
		b = tf.get_variable('b', [out_dim], 'float32',initializer=tf.constant_initializer(0.01))
		return tf.nn.bias_add(var, b)


def deconv3d(name, tensor, ksize , hsize , outshape, stddev=0.01, stride=2, h_stride=1, padding='SAME'):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [ksize, ksize,hsize, outshape[-1], tensor.get_shape()[-1]], dtype=tf.float32,
							initializer=tf.random_normal_initializer(stddev=stddev))
		var = tf.nn.conv3d_transpose(tensor, w, outshape, strides=[1, stride, stride, h_stride, 1], padding=padding)
		b = tf.get_variable('b', [outshape[-1]], 'float32', initializer=tf.constant_initializer(0.01))
		return tf.nn.bias_add(var, b)


def fully_connected(name,value, output_shape):
	with tf.variable_scope(name, reuse=None) as scope:
		shape = value.get_shape().as_list()
		w = tf.get_variable('w', [shape[1], output_shape], dtype=tf.float32,
									initializer=tf.random_normal_initializer(stddev=0.02))
		b = tf.get_variable('b', [output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

		return tf.matmul(value, w) + b

def lrelu(tensor):
	return tf.nn.leaky_relu( tensor, alpha=0.2)

class batch_norm(object):
	def __init__(self, name,epsilon=1e-5, momentum = 0.9):
		with tf.variable_scope(name):
			self.epsilon  = epsilon
			self.momentum = momentum
			self.name = name

	def __call__(self, x, train=True):
		return tf.contrib.layers.batch_norm(x,
			decay=self.momentum, 
			updates_collections=None,
			epsilon=self.epsilon,
			scale=True,
			is_training=train,
			scope=self.name)