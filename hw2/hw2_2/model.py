import numpy as np 
import tensorflow as tf 
from tensorflow.contrib import rnn,seq2seq

class Chatbot(object):
	"""docstring for Chatbot"""
	def __init__(self, word_dim):
		self.layer=2
		self.lr = 0.001
		self.word_dim = word_dim
		self.embedding_size = 1000
		self.hidden_units = 256
		self.sequence_length = 15
		self.beamsearch = False
		self.bidirectional = True
		self.initializer = tf.contrib.layers.xavier_initializer()
		self.embedding_layer = tf.get_variable(name='embedding',
			shape=[self.word_dim, self.embedding_size],initializer=self.initializer)

		self.__graph__()

	def __graph__(self):
		self.init_placeholder()
		self.build_encoder()
		self.build_decoder()
		self.build_optimizer()

	def init_placeholder(self):
		self.X = tf.placeholder(tf.int32, [None, self.sequence_length])
		self.Y = tf.placeholder(tf.int32, [None, self.sequence_length])
		self.dropout = tf.placeholder(tf.float32, [])
		self.batch_size = tf.placeholder(tf.int32, [])
		self.mask = tf.placeholder(tf.float32, [None, None])
		self.sp = tf.placeholder(tf.float32, [])
		self.len = tf.placeholder(tf.int32, [None])

	def build_encoder(self):
		# print ("build encoder ....... ")
		with tf.variable_scope('encoder'):

			self.encoder_inputs = tf.nn.embedding_lookup(self.embedding_layer, self.X)
			# projection_layer = tf.layers.Dense(self.hidden_units, use_bias=False, kernel_initializer=self.initializer)
			# self.encoder_inputs = projection_layer(self.encoder_inputs)

			if self.bidirectional == True:
				encoder_fw_cell = self.build_single_cell(self.hidden_units)
				encoder_bw_cell = self.build_single_cell(self.hidden_units)
				encoder_outputs, self.encoder_last_state = tf.nn.bidirectional_dynamic_rnn(
					encoder_fw_cell, 
					encoder_bw_cell, 
					self.encoder_inputs,
					dtype=tf.float32)
				self.encoder_outputs = tf.concat(encoder_outputs, -1)
			else:
				self.encoder_cell = rnn.MultiRNNCell([self.build_single_cell(self.hidden_units) for _ in range(self.layer)], state_is_tuple=True)

				self.encoder_outputs, self.encoder_last_state = tf.nn.dynamic_rnn(
					cell=self.encoder_cell, inputs=self.encoder_inputs, dtype=tf.float32, time_major=False)


	def build_decoder(self):
		# print ("build decoder ....... ")
		with tf.variable_scope('decoder'):
			# build cell list & attention
			cell_list = [self.build_single_cell(self.hidden_units) for i in range(self.layer)]
			# input_layer = tf.layers.Dense(self.hidden_units, name='input_projection')
			output_layer = tf.layers.Dense(self.word_dim, use_bias=False, kernel_initializer=self.initializer, name='output_projection')
			self.decoder_inputs = tf.nn.embedding_lookup(self.embedding_layer, self.Y)
			# self.decoder_inputs = input_layer(self.decoder_inputs)
			attention_mechanism = seq2seq.BahdanauAttention(
				num_units=self.hidden_units, 
				memory=self.encoder_outputs,normalize=True)
			cell_list[-1] = tf.contrib.seq2seq.AttentionWrapper(
				cell=cell_list[-1], 
				attention_mechanism=attention_mechanism,
				attention_layer_size=self.hidden_units,
				initial_cell_state=self.encoder_last_state[-1],
				name = 'attention_wrapper',
			)

			decoder_cell = rnn.MultiRNNCell(cells=cell_list, state_is_tuple=True)
			# RNN initial state
			initial_state = [state for state in self.encoder_last_state]
			initial_state[-1] = cell_list[-1].zero_state(batch_size=self.batch_size , dtype=tf.float32)
			decoder_initial_state = tuple(initial_state)
			self.decoder_initial_state = decoder_initial_state

			# training helper (training)
			training_helper = seq2seq.ScheduledEmbeddingTrainingHelper(
				inputs = self.decoder_inputs,	
				sequence_length = self.len,
				embedding = self.embedding_layer,
				sampling_probability=self.sp)

			training_decoder = seq2seq.BasicDecoder(
				cell=decoder_cell, 
				helper=training_helper, 
				initial_state=decoder_initial_state,
				output_layer=output_layer)
			decoder_outputs, decoder_state, decoder_length = seq2seq.dynamic_decode(training_decoder)
			self.logit = decoder_outputs.rnn_output
			self.loss = tf.contrib.seq2seq.sequence_loss(
				logits=self.logit,
				targets=self.Y,
				weights=self.mask)

	def build_optimizer(self):
		optimizer = tf.train.AdamOptimizer(self.lr)
		# self.train_op = optimizer.minimize(self.loss)
		gradients, variables = zip(*optimizer.compute_gradients(self.loss))
		gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
		self.train_op = optimizer.apply_gradients(zip(gradients, variables))
		
	def train(self, sess, X, Y, mask, batch_size, sp):
		sequence_length = [self.sequence_length]*batch_size
		l, _ = sess.run([self.loss, self.train_op], 
					feed_dict={self.X:X, self.Y:Y, 
					self.len:sequence_length, self.sp:sp, 
					self.batch_size:batch_size, self.dropout:0.1,
					self.mask: mask,
					})
		return l

	def test(self, sess, X, batch):
		Y = np.copy(X)
		slen = [self.sequence_length]*batch
		cur = sess.run(self.logit, feed_dict={self.X:X, self.Y:Y,
						self.len:slen, self.sp:0.8,
						self.batch_size:batch, self.dropout:0, })
		# cur = sess.run(self.infer_outputs, feed_dict={
		# 	self.X:X,self.batch_size:1, self.dropout:0, })
		return cur

	def build_single_cell(self, x):
		cell = rnn.LSTMCell(x, use_peepholes=True, initializer=tf.keras.initializers.glorot_normal(seed=7122))
		cell = rnn.DropoutWrapper(cell, 
			input_keep_prob=1-self.dropout, 
			output_keep_prob=1-self.dropout,
			state_keep_prob=1-self.dropout,
		)
		return cell

	def save(self, sess, path):
		saver = tf.train.Saver()
		saver.save(sess, save_path=path)

	def restore(self, sess, path):
		saver = tf.train.Saver()
		saver.restore(sess, save_path=path)





