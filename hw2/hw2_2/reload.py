import sys
import random
import pickle
import numpy as np
import tensorflow as tf 
from model import Chatbot
import argparse

parser = argparse.ArgumentParser(description='MLDS HW2-2')    
parser.add_argument('--data_loader', type=str, help='data loader class object')
parser.add_argument('-m', '--model', type=str, help='save model path')
parser.add_argument('--test', type=str, help='testing data')
parser.add_argument('--out', type=str, help='output file')
args = parser.parse_args()

test_file = args.test
fout = open(args.out, 'w')

with open(args.data_loader, 'rb') as f:
	data = pickle.load(f)

data.load_test(test_file)
batch = 50

with tf.Session() as sess:
	model = Chatbot(word_dim=data.word_dim)
	sess.run(tf.global_variables_initializer())
	model.restore(sess, args.model)
	index = 0
	for idx in range(data.test_len//batch+1):
		X = data.predict_batch(batch)
		X = np.array(X)
		bt = X.shape[0]
		pred = model.test(sess, X, bt)
		for idx in pred:
			onehot = np.argmax(idx, axis=1)
			sen = data.predict_output(onehot)
			sen = data.final_processing(sen, index)
			print (sen, file=fout)
			index += 1
			fout.flush()

