import os
import sys
import pickle
import numpy as np
import tensorflow as tf 
from data_v5 import Data
from model import Chatbot
import argparse
input_file = "../input/clr_conversation.txt"
data_file = "../model/data_class"
test_file = "../evaluation/test_input.txt"

parser = argparse.ArgumentParser(description='MLDS HW2-2')    
parser.add_argument('--data_loader', type=str, help='data loader')
parser.add_argument('-m', '--model', type=str, help='save model path')
parser.add_argument('--log', type=str, help='log file')
args = parser.parse_args()

# data = Data(input_file)
# with open(data_file, 'wb') as f:
# 	pickle.dump(data, f)

with open(args.data_loader, 'rb') as f:
	data = pickle.load(f)

# data.load_test(args.test)

iteration = 200000
bt = [32, 64]

path = os.path.join(args.model, "checkpoint.ckpt")
flog = open(args.log, 'w')

# flog = sys.stdout
with tf.Session() as sess:
	model = Chatbot(word_dim=data.word_dim)
	# model.restore(sess, "../model/iteration_0_/checkpoint.ckpt")
	# print ("reload success")
	sess.run(tf.global_variables_initializer())
	for _ in range(iteration):
		if _ < 100000:
			batch_size = bt[0]
		else:
			batch_size = bt[1]
		X, Y = data.next_batch(batch_size)
		X = np.array(X).reshape(batch_size, -1).astype(int)
		Y = np.array(Y).reshape(batch_size, -1).astype(int)
		mask = (Y != data.D['<pad>'])
		sp = max(0.3, 0.7 * (_ / iteration))
		loss = model.train(sess, X, Y, mask, batch_size, sp)

		if _ % 500 == 0:
			print ("iteration = ", _, "loss = ", loss, file=flog)
			pred = model.test(sess, np.array(X[0].reshape(1, -1)), 1)
			# pred = model.test(sess, np.array(X[0].reshape(1, -1)))
			# onehot = pred[0].reshape(-1)
			onehot = np.argmax(pred[0], axis=1)
			print ("Ques: ", data.one_to_sen(X[0]), file=flog)
			print ("Pred: ", data.one_to_sen(onehot), file=flog)
			print ("YAns: ", data.one_to_sen(Y[0]), file=flog)
			flog.flush()
		if _ % 10000 == 9999 and _ > 100000:
			
			model.save(sess, path)

