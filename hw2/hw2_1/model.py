import tensorflow as tf
from tensorflow.contrib import rnn,seq2seq
from tensorflow.contrib.training import HParams 

class im2txt:
	def __init__(self,hparams):


		#self.initialize = tf.contrib.layers.xavier_initializer()
		#self.initialize = tf.initializers.orthogonal(seed=7122)
		self.initialize  = tf.keras.initializers.glorot_normal(seed=7122)
		self.cell_initializer = tf.initializers.orthogonal(seed=7122)
		self.cell_initializer = self.initialize
		

		using_LSTM = True if hparams.cell_mode == 'LSTM' else False
		encoder_layers = hparams.encoder_layers
		decoder_layers = hparams.decoder_layers
		Activate_Bidirectional = True if hparams.is_bidirectional == 1 else False
		print("Activate_Bidirectional" , Activate_Bidirectional)
		def my_rnn_cell(x , needs_dropout = True , needs_residual = False , first_layer_dim = None):
			if using_LSTM:
				cell = rnn.LSTMCell(x,use_peepholes=True,initializer=self.cell_initializer)
			else:
				cell = rnn.GRUCell(x, kernel_initializer=self.cell_initializer)
			if needs_dropout:
				if first_layer_dim is None:
					# usual dropout
					cell = rnn.DropoutWrapper(cell, 
						input_keep_prob=1-hparams.dropout_rate, 
						output_keep_prob=1-hparams.dropout_rate,
						state_keep_prob=1-hparams.dropout_rate,
					)	
				else:
					# variational_recurrent dropour
					cell = rnn.DropoutWrapper(cell, 
					input_keep_prob=1-hparams.dropout_rate, 
					output_keep_prob=1-hparams.dropout_rate,
					state_keep_prob=1-hparams.dropout_rate,
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

		if Activate_Bidirectional:
			encoder_fw_layer_list = [my_rnn_cell(hparams.hidden_units) for _ in range(encoder_layers)]
			encoder_bw_layer_list = [my_rnn_cell(hparams.hidden_units) for _ in range(encoder_layers)]
			encoder_fw_cell = rnn.MultiRNNCell(encoder_fw_layer_list, state_is_tuple=True)
			encoder_bw_cell = rnn.MultiRNNCell(encoder_bw_layer_list, state_is_tuple=True)
			encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
				encoder_fw_cell, 
				encoder_bw_cell, 
				hparams.encoder_inputs,
				dtype=tf.float32)
			encoder_outputs = tf.concat(encoder_outputs, axis=2)
			encoder_state = tf.concat(encoder_state, axis=2)

		else:
			encoder_layer_list = [my_rnn_cell(hparams.hidden_units) for _ in range(encoder_layers)]
			encoder_cell = rnn.MultiRNNCell(encoder_layer_list, state_is_tuple=True)
			encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
				encoder_cell, 
				hparams.encoder_inputs,
				dtype=tf.float32)

		#--------------decoder layers-------------#

		decoder_layer_list = [my_rnn_cell(hparams.hidden_units) for _ in range(decoder_layers)]
		decoder_cell = rnn.MultiRNNCell(decoder_layer_list, state_is_tuple=True)
		
		#-------------attention layers------------#
		#attention_mechanism = seq2seq.LuongAttention(
		attention_mechanism = seq2seq.BahdanauAttention(
			num_units=hparams.hidden_units, 
			memory=encoder_outputs, normalize=True)
		#,scale = True)

		attention_decoder = seq2seq.AttentionWrapper(
			decoder_cell, attention_mechanism, output_attention=True,
			name = 'attention_wrapper')

		if not Activate_Bidirectional:
			decoder_initial_state = attention_decoder.zero_state(hparams.batch_size,tf.float32).clone(cell_state = encoder_state)
		else:
			decoder_initial_state = attention_decoder.zero_state(dtype=tf.float32,batch_size = hparams.batch_size)
		#-----------embedding layers--------------#
		embedding_matrix = tf.get_variable("embedding_matrix", [hparams.dim_vocabulary, hparams.embedding_size])
		decoder_emb_inputs = tf.nn.embedding_lookup(embedding_matrix, hparams.decoder_inputs)
		projection_layer = tf.layers.Dense(hparams.dim_vocabulary, use_bias=False, kernel_initializer=self.initialize)

		#-------------training decoder------------#
		training_helper = seq2seq.ScheduledEmbeddingTrainingHelper(
			inputs = decoder_emb_inputs,	
			sequence_length = hparams.sen_max_length,
			embedding = embedding_matrix,
			sampling_probability = hparams.sampling_prob)
		
		training_decoder = seq2seq.BasicDecoder(
			cell=attention_decoder, 
			helper=training_helper, 
			initial_state = decoder_initial_state,
			output_layer=projection_layer)
		
		decoder_outputs, decoder_state  , decoder_length = seq2seq.dynamic_decode(training_decoder)
		logits = decoder_outputs.rnn_output
		
		#---------------infer decoder-------------#
		if not Activate_Bidirectional:
			decoder_initial_state = attention_decoder.zero_state(hparams.batch_size * hparams.beam_width,tf.float32).clone(cell_state = encoder_state)
		else:
			decoder_initial_state = attention_decoder.zero_state(dtype=tf.float32, batch_size=hparams.batch_size * hparams.beam_width)
				
		# tiled_encoder_state = seq2seq.tile_batch( encoder_state , beam_width )
		# decoder_initial_state = decoder_initial_state.clone(
		# cell_state=tiled_encoder_state)
		ModeBeamSearch = True
		if ModeBeamSearch:
			infer_decoder = seq2seq.BeamSearchDecoder(
				cell = attention_decoder,
				embedding=embedding_matrix,
				start_tokens= tf.fill([hparams.batch_size],hparams.start_tokens),
				end_token=hparams.end_token,
				initial_state=decoder_initial_state,			
				beam_width=hparams.beam_width,
				output_layer=projection_layer, 
				length_penalty_weight=1.
			)
		else: 
			#Greedy Search
			infer_helper = seq2seq.GreedyEmbeddingHelper(
				embedding_matrix, 
				start_tokens=tf.fill([hparams.batch_size],hparams.start_tokens),
				end_token=hparams.end_token)
			

			infer_decoder = seq2seq.BasicDecoder(
				cell=attention_decoder, 
				helper=infer_helper,
				initial_state=decoder_initial_state, 
				output_layer=projection_layer)
		
		infer_outputs , infer_state , infer_length = seq2seq.dynamic_decode(infer_decoder,maximum_iterations=hparams.max_iterations)
		#--------------define loss-----------------#
		print(hparams.decoder_inputs)
		loss = tf.contrib.seq2seq.sequence_loss(
			logits=logits,
			targets=hparams.decoder_inputs,
			weights=hparams.loss_weight
		)
		optimizer = tf.train.AdamOptimizer(0.001)
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
			self.scores = infer_outputs.beam_search_decoder_output.scores
		else:
			self.infer_outputs = infer_outputs.predicted_ids
		
