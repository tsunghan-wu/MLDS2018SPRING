# coding=utf-8
import re
import sys
import random
import numpy as np
import pandas as pd

dic_file = "./input/dictionary.txt"
test_file = "../evaluation/input.txt"
input_file = "../input/clr_conversation.txt"

def is_ascii(s):
	return all(ord(c) < 128 for c in s)

class Data:
	def __init__(self, path):
		'''
		public:
			// bos = 0, eos = 1, pad = 2, unk = 3
			self.data: 2-dim array (conversation, sentence)
			self.word_dim: vocabulary
			self.D: dictionary (word2vec)
			self.inv_D: reverse dictionary (vec2word)
			self.N: # of training data
			self.next_batch_counter: batch counter
			self.sentence_length: padding to fixed length
		papameter:
			shortest conversation (conversation length)
			word_min_frequency 

		ps. I remove <55951 -> 56311> conversation because it is bad
		'''

		dic = pd.read_csv(dic_file, encoding='utf-8', sep='\s+', names=['index', 'char']).drop(['index'], axis=1)
		dic = dic['char'].tolist()
		dic = [x.encode('utf-8') for x in dic]
		dictionary = {}
		for idx, x in enumerate(dic):
			dictionary.update({x:idx})


		shortest_conversation = 2
		word_min_frequency = 10
		with open (path, 'r', encoding='utf-8') as f:
			raw_data = f.read()
		bag = raw_data.split("+++$+++")
		bag = [x for x in bag if len(x) >= shortest_conversation]

		data = []
		for x in bag:
			x = x.replace("　", " ")
			s = x.split('\n')
			data.append(s)

		self.word_dim = 0
		word_counter = np.zeros((20000000))
		self.D = {}
		for voc in ['<bos>' , '<eos>' , '<pad>' , '<unk>']:
			self.D.update({voc:self.word_dim})
			word_counter[self.D[voc]] = 99999
			self.word_dim += 1
		tmpdata = []
		for x in data:
			newbag = []
			for sen in x:
				news = []
				trash = [".", ")", "'", "*", "\"", "．", "]", "^", "`", "\\", "【", "】", "「", "」", "『", "』", "—", ",", "。"]
				for x in trash:
					sen = sen.replace(x, "")
				sen = re.sub('[\s]+', ' ', sen)
				s = sen.split(' ')
				for char in s:
					valid = True
					QQ = "莪"
					for c in char:
						if c.encode('utf-8') in dictionary or is_ascii(c):
							pass
						else:
							valid = False
							break
					if valid:
						if char.encode('utf-8') == QQ.encode('utf-8'):
							news.append("我")
						else:
							news.append(char)
						if char not in self.D:
							if char != ' ':
								self.D.update({char:self.word_dim})
							self.word_dim += 1
						word_counter[self.D[char]] += 1
				newsen = " ".join(news)
				newbag.append(newsen)
			tmpdata.append(newbag)

		self.D = {i:self.D[i] for i in self.D if word_counter[self.D[i]] >= word_min_frequency}
		self.word_dim = len(self.D)
		self.D = {i:j for i, j in zip(self.D, [_ for _ in range(self.word_dim)])}
		self.inv_D = { self.D[key] : key for key in self.D}
		self.data = []
		self.sentence_length = 15
		for x in tmpdata:
			bt = []
			for sen in x:
				news = ['<bos>']
				s = sen.split(' ')
				for char in s:
					if char not in self.D:
						news.append('<unk>')
					else:
						news.append(char)
				news.append('<eos>')
				new_sen = " ".join(news)
				bt.append(new_sen)
			self.data.append(bt)
		X = []
		Y = []
		for x in self.data:
			for sen in x[:-1]:
				X.append(sen)
			for sen in x[1:]:
				Y.append(sen)
		self.trainX = []
		self.trainY = []
		for x, y in zip(X, Y):
			tmpx = x.split(' ')
			tmpy = y.split(' ')
			if 3 < len(tmpx) <= self.sentence_length and 3 < len(tmpy) <= self.sentence_length:
				if tmpx.count('<unk>') < (len(tmpx)-2) / 3 and tmpy.count('<unk>') < (len(tmpy)-2) / 3:
					self.trainX.append(x)
					self.trainY.append(y)

		# print (len(self.trainX), len(self.trainY))
		self.N = len(self.trainX)
		self.next_batch_counter = 0
#        for x in self.D:
#            print ("word = ", x, "happens = ", word_counter[self.D[x]])

	def get_single_batch(self, idx):
		x = self.trainX[idx].split(' ')
		y = self.trainY[idx].split(' ')
		xx = []
		yy = []
		for word in x:
			xx.append(self.D[word])
		for word in y:
			yy.append(self.D[word])
		xx += [ self.D['<pad>'] for _ in range(self.sentence_length-len(xx))]
		yy += [ self.D['<pad>'] for _ in range(self.sentence_length-len(yy))]

		return xx, yy

	def next_batch(self, batch_size):
		max_len = 0
		X = []
		Y = []
		for _ in range(batch_size):
			x, y = self.get_single_batch(self.next_batch_counter)
			self.next_batch_counter += 1
			if self.next_batch_counter >= self.N:
				self.next_batch_counter = 0
			X.append(x)
			Y.append(y)
		return X, Y
	def one_to_sen(self,one_hot):
		return  ' '.join([ self.inv_D[idx] for idx in one_hot])

	def load_test(self, test_file):
		with open (test_file, 'r', encoding='utf-8') as f:
			raw_data = f.read()
		test_data = raw_data.split("\n")
		self.raw_test_data = test_data
		self.test_data = []
		for x in test_data:
			seg_list = x.split(' ')[:self.sentence_length-2]
			news = ['<bos>']
			for char in seg_list:
				if char not in self.D:
					news.append('<unk>')
				else:
					news.append(char)
			news.append('<eos>')
			self.test_data.append(" ".join(news))

		self.test_len = len(self.test_data)
		self.test_counter = 0


	def predict_batch(self, batch):
		X = []
		if self.test_len - self.test_counter < batch:
			batch_size = self.test_len - self.test_counter
		else:
			batch_size = batch
		for _ in range(batch_size):
			x = []
			for word in self.test_data[self.test_counter].split():
				x.append(self.D[word])
			x += [self.D['<pad>'] for _ in range(self.sentence_length-len(x))]
			self.test_counter += 1
			X.append(x)
		return X

	def predict_output(self, s):
		news = []
		start = 0
		if s[0] == 0:
			start = 1
		else:
			start = 0
		for x in s[start:]:
			if x == 1:
				break
			elif x == 2:
				pass
			else:
				news.append(x)
		out = []
		if len(news) == 0:
			return ""
		out.append(news[0])
		for x in range(1, len(news)):
			if ((news[x] != news[x-1]) and (news[x] != '<unk>')):
				out.append(news[x])
		sen = self.one_to_sen(out)
		return sen

	def final_processing(self, sen, index):
		tmp = sen.split(' ')
		r = random.random()
		if sen == ("") or sen == ("我"):	
			return self.raw_test_data[index]
		elif len(tmp) > 8:
			if r < 0.8:
				tmp = tmp[:8]
				sen = " ".join(tmp)
				return sen
			else:
				return self.raw_test_data[index]
		elif "我" in sen and len(tmp) <= 2:
			if r < 0.5:
				return sen
			else:
				return self.raw_test_data[index]
		else:
			return sen

		
if __name__ == '__main__':
	training_data = Data(input_file)
	print (training_data.N)
	print (len(training_data.D))

	exit()
	training_data.test(test_file)



