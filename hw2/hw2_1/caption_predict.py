import tensorflow as tf
import numpy as np
import math
import sys
import pickle
from tensorflow.contrib import rnn,seq2seq
# sys.path.append('drive/deep/hw3/')
from video_data import caption_test
from video_data_v2 import caption_data
#from raw_spliting import caption_data
from model import im2txt
from tensorflow.contrib.training import HParams 

### all params ###
config = {}
outfile_path = sys.argv[1]

def read_config(config_filename):

	print('{0:-^40s}'.format('reading_config'))
	print('config_path:' , config_filename)
	for row in open(config_filename,'r'):
		print(row)
		if row[0] == '#':
			continue
		row = row.split()
		try:
			config.update({row[0]:int(row[-1])})
		except:
			try:
				config.update({row[0]:float(row[-1])})
			except:
				config.update({row[0]:row[-1]})
	print('{0:-^40s}'.format('reading_config'))
	
	
if __name__ == '__main__':
	'''
	data must given:
		data.D 
		data.word_dim
		data.next_batch
		data.singel_test
		data.one_to_sen
	'''
	data = pickle.load(open('processed_training_data_v2','rb'))
	#data = pickle.load(open('raw_training_data_v2','rb'))
	test_data = pickle.load(open('reduce_test_data','rb'))
	print('word_dim : ', data.word_dim)
	read_config('model.config')
	
	timesteps = 80
	dim_feat = 4096

	iteration = config['iteration']
	train_bt = config['batch_size']
	beam_width = config['beam_width']

	batch_size = tf.placeholder(tf.int32 , [], name='batch_size')
	dropout_rate = tf.placeholder(tf.float32 , [])
	sampling_prob = tf.placeholder(tf.float32 , [], name='schedule_sampling_prob')
	
	X = tf.placeholder(tf.float32, [None, timesteps,dim_feat], name='input_batch_feat')
	placeholder_beam_width = tf.placeholder(tf.int32 , [])
	Xtrans = seq2seq.tile_batch(X,placeholder_beam_width)
	
	Y = tf.placeholder(tf.int32, [None,None], name='input_caption')	
	sequence_length = tf.placeholder(tf.int32,[None])
	loss_weight = tf.placeholder(tf.float32 , [None,None])
	
	hparams = HParams(
			encoder_inputs=Xtrans,
			decoder_inputs=Y,
			hidden_units=config['hidden_units'],
			dropout_rate=dropout_rate,
			dim_vocabulary=data.word_dim,
			sen_max_length=sequence_length,
			embedding_size=config['embedding_size'],
			sampling_prob=sampling_prob,
			beam_width=beam_width,
			batch_size=batch_size,
			start_tokens=data.D['<bos>'],
			end_token=data.D['<eos>'],
			loss_weight=loss_weight,
			max_iterations=config['max_iteration'],
			cell_mode=config['cell_mode'],
			encoder_layers=config['encoder_layers'],
			decoder_layers=config['decoder_layers'],
			is_bidirectional = config['is_bidirectional']
		)
	model = im2txt(hparams)
	saver = tf.train.Saver()
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		saver.restore(session, "final4_model/model.ckpt")
				
		### predict ###
		y_pred = []
		filename = [x[:-4] for x in test_data.file_lists]
		for feat, name in zip(test_data.test_data, filename):
			predict,scores = session.run([model.infer_outputs,model.scores], {
				X: [feat] ,
				placeholder_beam_width : beam_width,
				batch_size : 1,
				dropout_rate : 0
			})
			print('{:-^40s}'.format(name))
			if beam_width == 1:
				result = data.predict(predict[1].reshape(-1))
			else:
				result = predict[:,:,0].reshape(-1)
				new_result = [result[0]]
				if len(result) >= 1:
					for iidx in range(1,len(result)):
						if result[iidx] != new_result[-1]:
							new_result.append(result[iidx])
				result = data.predict(new_result)
			print ('{: ^12s}'.format(str(np.sum(scores[0],0)[0])),result)
			y_pred.append(result)

		import csv
		output_file = outfile_path
		#output_file = config['result_path'] + '/output_' + str(each_iteration) +'.txt'
		with open(output_file, 'w') as f:
			writer = csv.writer(f, delimiter=',')
			writer.writerows(zip(filename, y_pred))
		
