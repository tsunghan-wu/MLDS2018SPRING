import tensorflow as tf
from tensorflow.contrib import rnn,seq2seq

class S2VT(object):
	"""docstring for S2VT"""
	def __init__(self, video_dim, video_timestep, word_dim, hidden_units, batch_size, beam_width, maxiter, start_tokens, end_token):
		self.video_dim = video_dim
		self.word_dim = word_dim
		self.hidden_units = hidden_units
		self.video_timestep = video_timestep
		self.batch_size = batch_size
		self.beam_width = beam_width
		self.maxiter = maxiter
		self.start_tokens = start_tokens
		self.end_token = end_token
		self.initialize = tf.contrib.layers.xavier_initializer()
	
	def gru_cell(self):
		gru = rnn.GRUCell(self.hidden_units, kernel_initializer=self.initialize)
		return gru

	def attention_method(self, enc_out):
		att = tf.transpose(enc_out, [0, 1, 2])
		attention_mechanism = seq2seq.LuongAttention(self.hidden_units, memory=enc_out)
		attention_cell = tf.contrib.seq2seq.AttentionWrapper(
			cell=self.gru_cell(), 
			attention_mechanism=attention_mechanism,
			attention_layer_size=self.hidden_units,
			name = 'attention_wrapper')
		return attention_cell	


	def train(self):
		### place holder ###
		with tf.variable_scope("inputs"):
			video = tf.placeholder(tf.float32, [None, self.video_timestep, self.video_dim])
			caption = tf.placeholder(tf.int32, [self.batch_size,None])
			caption_shape = tf.placeholder(tf.int32,[2])
			sequence_length = tf.placeholder(tf.int32,[None])
			sampling_prob = tf.placeholder(tf.float32 , [], name='schedule_sampling_prob')

		### 2 layer encoder + attention ###
		with tf.variable_scope("RNN"):
			cell = rnn.MultiRNNCell([self.gru_cell() for _ in range(2)], state_is_tuple=True)
			encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, video, dtype=tf.float32)
			### 2 layer decoder ###
			attention_cell = self.attention_method(encoder_output)
			decoder_cell = rnn.MultiRNNCell(cells=[attention_cell, self.gru_cell()], state_is_tuple=True)
			### embedding words ###
			embedding_matrix = tf.get_variable("embedding_matrix", [self.word_dim, self.hidden_units])
			decoder_inputs = tf.nn.embedding_lookup(embedding_matrix, caption)
			projection_layer = tf.layers.Dense(self.word_dim, use_bias=False, kernel_initializer=self.initialize)

			### Scheduled Sampling Decoder ###
			training_helper = seq2seq.ScheduledEmbeddingTrainingHelper(
				inputs = decoder_inputs,	
				sequence_length = sequence_length,
				embedding = embedding_matrix,
				sampling_probability = sampling_prob)

			training_decoder = seq2seq.BasicDecoder(
				decoder_cell, 
				training_helper, 
				decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size),
				##??????????## encode_state?
				output_layer=projection_layer)

		with tf.variable_scope("output"):
			decoder_outputs, decoder_state, decoder_length= seq2seq.dynamic_decode(training_decoder)
		
		### Loss function ###	
		with tf.variable_scope("cost"):
			self.logits = decoder_outputs.rnn_output
			self.loss = tf.contrib.seq2seq.sequence_loss(
				logits=self.logits,
				targets=caption,
				weights=tf.ones([self.batch_size, sequence_length[0]]))

		with tf.variable_scope("train"):
			self.train_op = tf.train.AdamOptimizer(0.001).minimize(self.loss)
		
		return video, caption, sequence_length, sampling_prob

	def test(self):
		with tf.variable_scope("inputs"):
			video = tf.placeholder(tf.float32, [None, self.video_timestep, self.video_dim])
			caption = tf.placeholder(tf.int32, [1,None])
			caption_shape = tf.placeholder(tf.int32,[2])
			sequence_length = tf.placeholder(tf.int32,[None])
			sampling_prob = tf.placeholder(tf.float32 , [], name='schedule_sampling_prob')

		### 2 layer encoder + attention ###
		with tf.variable_scope("RNN") as scope:
			scope.reuse_variables()
			cell = rnn.MultiRNNCell([self.gru_cell() for _ in range(2)], state_is_tuple=True)
			encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, video, dtype=tf.float32)
			### 2 layer decoder ###
			attention_cell = self.attention_method(encoder_output)
			decoder_cell = rnn.MultiRNNCell(cells=[attention_cell, self.gru_cell()], state_is_tuple=True)
			### embedding words ###
			embedding_matrix = tf.get_variable("embedding_matrix", [self.word_dim, self.hidden_units])
			decoder_inputs = tf.nn.embedding_lookup(embedding_matrix, caption)
			projection_layer = tf.layers.Dense(self.word_dim, use_bias=False, kernel_initializer=self.initialize)

			### Beam Search Decoder ###
			infer_decoder = seq2seq.BeamSearchDecoder(
				cell=decoder_cell,
				embedding=embedding_matrix,
				start_tokens= tf.fill([1],self.start_tokens),
				end_token=self.end_token,
				initial_state=seq2seq.tile_batch(encoder_state, multiplier=self.beam_width),
					# decoder_cell.zero_state(batch_size * beam_width, tf.float32).clone(
					# 	cell_state=seq2seq.tile_batch(encoder_state, beam_width)),
				beam_width=self.beam_width,
				output_layer=projection_layer, 
				#length_penalty_weight=0.0
			)

		with tf.variable_scope("output"):
			decoder_outputs, decoder_state, decoder_length= seq2seq.dynamic_decode(training_decoder)
			infer_outputs , infer_state , infer_length = seq2seq.dynamic_decode(infer_decoder,maximum_iterations=self.maxiter)
			self.infer_outputs = infer_outputs.predicted_ids

		return video

