import tensorflow as tf
from tensorflow.contrib import rnn,seq2seq

class im2txt:
	def __init__(self, parameters):
		### input ###
		self.encoder_inputs = parameters['encoder_inputs']

		decoder_inputs = parameters['decoder_inputs']
		decoder_inputs_shape = parameters['decoder_inputs_shape']
		sampling_prob = parameters['sampling_prob']

		### constant ###
		self.hidden_units=parameters['encode_num_hidden']
		self.dim_vocabulary=parameters['dim_vocabulary']
		sen_max_length = parameters['sentence_length']
		embedding_size = parameters['embedding_size']
		
		beam_width = parameters['beam_width']
		start_tokens = parameters['start_tokens']
		end_token = parameters['end_token']
		batch_size = parameters['batch_size']
		max_iterations = parameters['max_iterations']
		timesteps = self.encoder_inputs.shape[1]

		self.initialize = tf.contrib.layers.xavier_initializer()

		def lstm_cell(x):
			lstm = rnn.BasicLSTMCell(x)
			return lstm
		def gru_cell(x):
			gru = rnn.GRUCell(x, kernel_initializer=self.initialize)
			return gru
		def _batch_norm(x, mode='train', name=None):
			return tf.contrib.layers.batch_norm(inputs=x,decay=0.95,
												center=True,scale=True,
												is_training=(mode=='train'),
												updates_collections=None,
												scope=(name+'batch_norm'))


		# encoder_inputs = _batch_norm(encoder_inputs, mode=mode, name='conv_features')

		#-----------embedding layers--------------#
		embedding_matrix = tf.get_variable("embedding_matrix", [self.dim_vocabulary, embedding_size])
		
		decoder_emb_inputs = tf.nn.embedding_lookup(embedding_matrix, decoder_inputs)

		projection_layer = tf.layers.Dense(self.dim_vocabulary, use_bias=False, kernel_initializer=self.initialize)

		#--------------encoder layers-------------#

		## two layer LSTM encoding ##
		encoder_layers = 2
		# cell = tf.contrib.rnn.MultiRNNCell([gru_cell(self.hidden_units) for _ in range(encoder_layers)], state_is_tuple=True)
		# encoder_outputs, encoder_state = tf.nn.dynamic_rnn(cell, self.encoder_inputs, dtype=tf.float32)
		fw_cell = tf.contrib.rnn.MultiRNNCell([gru_cell(self.hidden_units//2) for _ in range(encoder_layers)], state_is_tuple=True)
		bw_cell = tf.contrib.rnn.MultiRNNCell([gru_cell(self.hidden_units//2) for _ in range(encoder_layers)], state_is_tuple=True)
		(encoder_outputs_fw, encoder_outputs_bw),(_,__) = \
			tf.nn.bidirectional_dynamic_rnn(
				fw_cell, 
				bw_cell,
				self.encoder_inputs, dtype=tf.float32
			)
		encoder_outputs = tf.concat([encoder_outputs_fw, encoder_outputs_bw], axis=2)

		#print(encoder_outputs_fw)
		#print(encoder_outputs_bw)
		#exit()
		
		# attention_states: [batch_size, max_time, self.hidden_units]
		attention_states = tf.transpose(encoder_outputs, [0, 1, 2])
		attention_mechanism = seq2seq.LuongAttention(
			self.hidden_units, attention_states , scale = True)

		#--------------decoder layers-------------#
			#---#
		decoder_layers = 2
		cell_list = [gru_cell(self.hidden_units) for i in range(decoder_layers)]

		attention_cell = cell_list.pop(0)
		

		attention_cell = tf.contrib.seq2seq.AttentionWrapper(
			attention_cell, attention_mechanism,
			attention_layer_size=self.hidden_units,
			name = 'attention_wrapper')

		decoder_cell = rnn.MultiRNNCell(cells=[attention_cell]+cell_list, state_is_tuple=True)

		#--------------training decoder------------#
		training_helper = seq2seq.ScheduledEmbeddingTrainingHelper(
			inputs = decoder_emb_inputs,	
			sequence_length = sen_max_length,
			embedding = embedding_matrix,
			sampling_probability = sampling_prob)
		infer_decoder = seq2seq.BeamSearchDecoder(
			cell=decoder_cell,
			embedding=embedding_matrix,
			start_tokens= tf.fill([batch_size],start_tokens),
			end_token=end_token,
			initial_state=decoder_cell.zero_state(batch_size * beam_width, tf.float32),
			beam_width=beam_width,
			output_layer=projection_layer, 
			length_penalty_weight=1.
		)
		#tf.div( (5. + tf.to_float(sequence_lengths))**penalty_factor, (5. + 1.)
		#        **penalty_factor)
		training_decoder = seq2seq.BasicDecoder(
			cell=decoder_cell, 
			helper=training_helper, 
			initial_state=decoder_cell.zero_state(dtype=tf.float32, batch_size=batch_size),
			output_layer=projection_layer)

		decoder_outputs, decoder_state  , decoder_length= seq2seq.dynamic_decode(training_decoder)
		infer_outputs , infer_state , infer_length = seq2seq.dynamic_decode(infer_decoder,maximum_iterations=max_iterations)
		
		#--------------define loss-----------------#
		logits = decoder_outputs.rnn_output
		print(decoder_inputs)
		loss = tf.contrib.seq2seq.sequence_loss(
			logits=logits,
			targets=decoder_inputs,
			weights=tf.ones(decoder_inputs_shape))
		train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

		#--------------release--------------------#
		self.logits = logits
		self.loss = loss
		self.train_op=train_op
		self.infer_outputs = infer_outputs.predicted_ids

