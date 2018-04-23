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
		dropout_rate =parameters['dropout_rate']
		sen_max_length = parameters['sentence_length']
		embedding_size = parameters['embedding_size']
		
		beam_width 	= parameters['beam_width']
		start_tokens = parameters['start_tokens']
		end_token 	= parameters['end_token']
		batch_size 	= parameters['batch_size']
		max_iterations = parameters['max_iterations']
		loss_weight = parameters['loss_weight']
		timesteps = self.encoder_inputs.shape[1]

		#self.initialize = tf.contrib.layers.xavier_initializer()
		#self.initialize = tf.initializers.orthogonal(seed=7122)
		self.initialize  = tf.keras.initializers.glorot_normal(seed=7122)
		self.cell_initializer = tf.initializers.orthogonal(seed=7122)
		self.cell_initializer = self.initialize
		using_LSTM = True
		def my_rnn_cell(x , needs_dropout = True , needs_residual = False , first_layer_dim = None):
			if using_LSTM:
				cell = rnn.LSTMCell(x,use_peepholes=True,initializer=self.cell_initializer)
			else:
				cell = rnn.GRUCell(x, kernel_initializer=self.cell_initializer)
			if needs_dropout:
				if first_layer_dim is None:
					# usual dropout
					cell = rnn.DropoutWrapper(cell, 
						input_keep_prob=1-dropout_rate, 
						output_keep_prob=1-dropout_rate,
						state_keep_prob=1-dropout_rate,
					)	
				else:
					# variational_recurrent dropour
					cell = rnn.DropoutWrapper(cell, 
					input_keep_prob=1-dropout_rate, 
					output_keep_prob=1-dropout_rate,
					state_keep_prob=1-dropout_rate,
					variational_recurrent =True,
					dtype=tf.float32,
					input_size = first_layer_dim
					)	
			
			if needs_residual:
				cell = rnn.ResidualWrapper(cell)
		
			return cell
		def _batch_norm(x, mode='train', name=None):
			return tf.contrib.layers.batch_norm(inputs=x,decay=0.95,
												center=True,scale=True,
												is_training=(mode=='train'),
												updates_collections=None,
												scope=(name+'batch_norm'))


		# encoder_inputs = _batch_norm(encoder_inputs, mode=mode, name='conv_features')

		#--------------encoder layers-------------#

		encoder_layers = 2
		Activate_Bidirectional = True
		if Activate_Bidirectional:
			encoder_fw_layer_list = [my_rnn_cell(self.hidden_units//2) for _ in range(encoder_layers)]
			encoder_bw_layer_list = [my_rnn_cell(self.hidden_units//2) for _ in range(encoder_layers)]
			encoder_fw_cell = rnn.MultiRNNCell(encoder_fw_layer_list, state_is_tuple=True)
			encoder_bw_cell = rnn.MultiRNNCell(encoder_bw_layer_list, state_is_tuple=True)
			encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
				encoder_fw_cell, 
				encoder_bw_cell, 
				self.encoder_inputs,
				dtype=tf.float32)
			encoder_outputs = tf.concat(encoder_outputs, axis=2)
			encoder_state = tf.concat(encoder_state, axis=2)

		else:
			encoder_layer_list = [my_rnn_cell(self.hidden_units) for _ in range(encoder_layers)]
			encoder_cell = rnn.MultiRNNCell(encoder_layer_list, state_is_tuple=True)
			encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
				encoder_cell, 
				self.encoder_inputs,
				dtype=tf.float32)

		#--------------decoder layers-------------#

		decoder_layers = 2
		decoder_layer_list = [my_rnn_cell(self.hidden_units) for _ in range(decoder_layers)]
		decoder_cell = rnn.MultiRNNCell(decoder_layer_list, state_is_tuple=True)
		
		#-------------attention layers------------#
		attention_mechanism = seq2seq.LuongAttention(
			num_units=self.hidden_units, 
			memory=encoder_outputs, scale = True)

		attention_decoder = seq2seq.AttentionWrapper(
			decoder_cell, attention_mechanism, output_attention=True,
			name = 'attention_wrapper')

		if not Activate_Bidirectional:
			decoder_initial_state = attention_decoder.zero_state(batch_size,tf.float32).clone(cell_state = encoder_state)
		else:
			decoder_initial_state = attention_decoder.zero_state(dtype=tf.float32,batch_size = batch_size)
		#-----------embedding layers--------------#
		embedding_matrix = tf.get_variable("embedding_matrix", [self.dim_vocabulary, embedding_size])
		decoder_emb_inputs = tf.nn.embedding_lookup(embedding_matrix, decoder_inputs)
		projection_layer = tf.layers.Dense(self.dim_vocabulary, use_bias=False, kernel_initializer=self.initialize)

		#-------------training decoder------------#
		training_helper = seq2seq.ScheduledEmbeddingTrainingHelper(
			inputs = decoder_emb_inputs,	
			sequence_length = sen_max_length,
			embedding = embedding_matrix,
			sampling_probability = sampling_prob)
		
		training_decoder = seq2seq.BasicDecoder(
			cell=attention_decoder, 
			helper=training_helper, 
			initial_state = decoder_initial_state,
			output_layer=projection_layer)
		
		decoder_outputs, decoder_state  , decoder_length = seq2seq.dynamic_decode(training_decoder)
		logits = decoder_outputs.rnn_output
		
		#---------------infer decoder-------------#
		if not Activate_Bidirectional:
			decoder_initial_state = attention_decoder.zero_state(batch_size * beam_width,tf.float32).clone(cell_state = encoder_state)
		else:
			decoder_initial_state = attention_decoder.zero_state(dtype=tf.float32, batch_size=batch_size * beam_width)
				
		# tiled_encoder_state = seq2seq.tile_batch( encoder_state , beam_width )
		# decoder_initial_state = decoder_initial_state.clone(
		# cell_state=tiled_encoder_state)
		ModeBeamSearch = True
		if ModeBeamSearch:
			infer_decoder = seq2seq.BeamSearchDecoder(
				cell = attention_decoder,
				embedding=embedding_matrix,
				start_tokens= tf.fill([batch_size],start_tokens),
				end_token=end_token,
				initial_state=decoder_initial_state,			
				beam_width=beam_width,
				output_layer=projection_layer, 
				length_penalty_weight=0.8
			)
		else: 
			#Greedy Search
			infer_helper = seq2seq.GreedyEmbeddingHelper(
				embedding_matrix, 
				start_tokens=tf.fill([batch_size],start_tokens),
				end_token=end_token)
			

			infer_decoder = seq2seq.BasicDecoder(
				cell=attention_decoder, 
				helper=infer_helper,
				initial_state=decoder_initial_state, 
				output_layer=projection_layer)
		
		infer_outputs , infer_state , infer_length = seq2seq.dynamic_decode(infer_decoder,maximum_iterations=max_iterations)
		#--------------define loss-----------------#
		print(decoder_inputs)
		loss = tf.contrib.seq2seq.sequence_loss(
			logits=logits,
			targets=decoder_inputs,
			weights=loss_weight
		)
		optimizer = tf.train.AdamOptimizer(0.0001)
		Apply_gradient_clipping = True
		if Apply_gradient_clipping:
			gradients, variables = zip(*optimizer.compute_gradients(loss))
			gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
			train_op = optimizer.apply_gradients(zip(gradients, variables))
		else:
			train_op = optimizer.minimize(loss)

		#--------------release--------------------#
		self.logits = logits
		self.loss = loss
		self.train_op=train_op
		if ModeBeamSearch:
			self.infer_outputs = infer_outputs.predicted_ids
		else:
			self.infer_outputs = infer_outputs.predicted_ids
		
