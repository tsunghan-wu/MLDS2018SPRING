import tensorflow as tf
import numpy as np
import math
import sys
from tensorflow.contrib import rnn,seq2seq
# sys.path.append('drive/deep/hw3/')
from video_data import caption_data , caption_test
from model import im2txt
##############################
from termcolor import colored#
##############################

if __name__ == '__main__':
	word_min_frequency = 1
	data = caption_data(word_min_frequency,verbose=True,extraction=True)
	test_data = caption_test()
	'''
		data has two method:
		X , Y = getASingleCap( index )
		[ (X , Y) ]* batchsize = next_batch ( batch_size )
	'''

	timesteps = 80
	dim_feat = 4096
	epoch = int(1e4)
	train_bt = 10

	# batch_size , timesteps , dim_feat
	X = tf.placeholder(tf.float32, [None, timesteps,dim_feat], name='input_batch_feat')
	# batch
	Y = tf.placeholder(tf.int32, [None,None], name='input_caption')
	sequence_length = tf.placeholder(tf.int32,[None])
	caption_shape = tf.placeholder(tf.int32,[2])
	batch_size = tf.placeholder(tf.int32 , [], name='batch_size')
	dropout_rate = tf.placeholder(tf.float32 , [])
	beam_width = 5
	placeholder_beam_width = tf.placeholder(tf.int32 , [])
	loss_weight = tf.placeholder(tf.float32 , [None,None])
	sampling_prob = tf.placeholder(tf.float32 , [], name='schedule_sampling_prob')
	Xtrans = seq2seq.tile_batch(X,placeholder_beam_width)
	model = im2txt(
		{
			'encoder_inputs': Xtrans,
			'decoder_inputs': Y,
			'decoder_inputs_shape': caption_shape,
			# bidirectional -> 256/256
			'encode_num_hidden' : 256,
			'dropout_rate' : dropout_rate,
			'dim_vocabulary' : data.word_dim,
			'sentence_length' : sequence_length,
			'embedding_size' : 1000,
			'sampling_prob' : sampling_prob,
			'beam_width' : beam_width,
			'batch_size' : batch_size,
			'start_tokens': data.D['<bos>'] ,
			'end_token': data.D['<eos>'],
			'loss_weight' : loss_weight,
			'max_iterations' : 15
		})

	saver = tf.train.Saver()
	with tf.Session() as session:
		session.run(tf.global_variables_initializer())
		#saver.restore(session, "model/model.ckpt")
				
		for iteration in range(epoch):
			# get next batch
			feature,caption = data.next_batch(train_bt)
			seq_len = len(caption[0])	# single caption's sentence_len
			# print("iter = ", iteration, "sentence_length = ", seq_len)
			# training 

			## sampling function: 1 - sigmoid(cur_iter/tot_iter)
			k= 1000
			sp =  0.5 + 0.5 *iteration / epoch
			#sp = min(0.8 + iteration/epoch*0.2,0.3 + (1 - k/(k+np.exp( (iteration+k)/k))))
			# print ("caption_shape = ", caption.shape)
			# print ("seq_len = ", [seq_len])
			_, _loss , predict = session.run([model.train_op, model.loss , model.logits], {
				X: feature,	# trainX
				Y: caption,	# trainY
				batch_size : train_bt,	# bt
				placeholder_beam_width : 1,
				caption_shape : caption.shape,
				sequence_length : [seq_len]*train_bt,
				dropout_rate : 0.3,
				sampling_prob : sp,
				loss_weight : caption != data.D['<pad>']
			})
			# print("sampling_prob = ", session.run(sampling_prob))
			print("iteration:%5d"%(iteration) , "Loss: %6g" %(_loss) , 'sp:%6g' %(sp))
			if iteration % 50 == 49:
				for each_batch in predict[0:10]:
					one_hot = np.argmax(each_batch,1)
					print(data.one_to_sen(one_hot))
					#print(one_hot)
					#print(each_batch.shape , one_hot.shape)
			if iteration % 50 == 4:
				feat , name = test_data.single_test()
				predict = session.run([model.infer_outputs], {
					X: [feat] ,
					placeholder_beam_width : beam_width,
					batch_size : 1,
					dropout_rate : 0
				})
				print('{:-^40s}'.format(name))
				if beam_width != 1:
					for case in range(beam_width):
						print('{0:-^40s}'.format('case' + str(case+1)))
						print(colored(data.one_to_sen(predict[0][:,:,case].reshape(-1)),'red'))
						print(colored(data.predict(predict[0][:,:,case].reshape(-1)),'cyan'))
				else:
					print(predict[0][0].shape)
					print(predict[0][0])
					predict = predict[0][1].reshape(-1)
					print(colored(data.one_to_sen(predict),'red'))
					print(colored(data.predict(predict),'cyan'))
				
			if iteration % 500 == 0:
				### predict ###
				y_pred = []
				filename = [x[:-4] for x in test_data.file_lists]
				for feat, name in zip(test_data.test_data, filename):
					predict = session.run([model.infer_outputs], {
						X: [feat] ,
						placeholder_beam_width : beam_width,
						batch_size : 1,
						dropout_rate : 0
					})
					print('{:-^40s}'.format(name))
					if beam_width == 1:
						result = data.predict(predict[0][1].reshape(-1))
					else:
						result = data.predict(predict[0][:,:,0].reshape(-1))
					print (result)
					y_pred.append(result)

				import csv
				output_file = '../output.txt'
				with open(output_file, 'w') as f:
					writer = csv.writer(f, delimiter=',')
					writer.writerows(zip(filename, y_pred))
			if iteration % 10000 == 0:
				save_path = saver.save(session, "model/model.ckpt")
				print(colored('model is saved in %s.'%(save_path),'yellow'))

