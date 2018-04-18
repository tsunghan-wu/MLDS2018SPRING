import tensorflow as tf
import numpy as np
import math
import sys
from tensorflow.contrib import rnn
# sys.path.append('drive/deep/hw3/')
from video_data import caption_data , caption_test
from s2vt import S2VT

if __name__ == '__main__':
	word_min_frequency = 2
	data = caption_data(word_min_frequency,verbose=True)
	test_data = caption_test()
	'''
		data has two method:
		X , Y = getASingleCap( index )
		[ (X , Y) ]* batchsize = next_batch ( batch_size )
	'''

	timesteps = 80
	dim_feat = 4096
	epoch = int(500)
	train_bt = 30
	encode_num_hidden = 300
	beam_width = 5

	# batch_size , timesteps , dim_feat
	model = S2VT(video_dim=4096, video_timestep=80, word_dim=data.word_dim, 
				hidden_units=300, batch_size=10, beam_width=5, maxiter=20, 
				start_tokens=data.D['<bos>'], end_token=data.D['<eos>'])

	train_video, train_caption, sequence_length, sampling_prob = model.train()
	tf_video = model.test()
	with tf.variable_scope("RNN") as scope:
		session = tf.Session()
		session.run(tf.global_variables_initializer())
		for iteration in range(epoch):
			# get next batch
			feature,caption = data.next_batch(model.batch_size)
			seq_len = len(caption[0])	# single caption's sentence_len
			sp = 1 / math.e**(iteration / epoch)

			_, _loss , predict = session.run([model.train_op, model.loss , model.logits], {
				train_video: feature,	# trainX
				train_caption: caption,	# trainY
				sequence_length : [seq_len]*model.batch_size,
				sampling_prob : sp,
			})
			# print("sampling_prob = ", session.run(sampling_prob))
			print("iteration:%5d"%(iteration) , "Loss: " + str(_loss))
			# if iteration % 50 == 49:
			# 	for each_batch in predict[0:10]:
			# 		one_hot = np.argmax(each_batch,1)
			# 		print(data.one_to_sen(one_hot))
			# 		#print(one_hot)
			# 		#print(each_batch.shape , one_hot.shape)
			if iteration % 10 == 0:
				feat , name = test_data.single_test()
				predict = session.run([model.infer_outputs], {
					tf_video: [feat] * beam_width,
				})
				print('{:-^40s}'.format(name))
				for case in range(beam_width):
					print('case:' , case+1 , data.one_to_sen(predict[0][:,:,case].reshape(-1)))
			if iteration % 100 == 99:
				### predict ###
				y_pred = []
				filename = [x[:-4] for x in test_data.file_lists]
				for feat, name in zip(test_data.test_data, filename):
					predict = session.run([model.infer_outputs], {
						tf_video: [feat] * beam_width,
					})
					print('{:-^40s}'.format(name))
					result = data.predict(predict[0][:,:,0].reshape(-1))
					print (result)
					y_pred.append(result)

				import csv
				output_file = '../MLDS_hw2_1_data/output.txt'
				with open(output_file, 'w') as f:
					writer = csv.writer(f, delimiter=',')
					writer.writerows(zip(filename, y_pred))


