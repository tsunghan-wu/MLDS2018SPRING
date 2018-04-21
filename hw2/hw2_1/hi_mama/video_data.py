import numpy as np
import pandas as pd
import json
import pickle
import os

label_path = 'MLDS_hw2_1_data/training_label.json'
feat_path  = 'MLDS_hw2_1_data/training_data/feat'
test_path  = 'MLDS_hw2_1_data/testing_data/feat'


# label_path = 'drive/deep/hw2/MLDS_hw2_1_data/training_label.json'
# feat_path  = 'drive/deep/hw2/MLDS_hw2_1_data/training_data/feat'
# test_path  = 'drive/deep/hw2/MLDS_hw2_1_data/testing_data/feat'
PAD = '<pad>'
		
def _removeNonAscii(s):
	return "".join([i for i in s if ord(i)<128])

word_collection = set()

mirror = pickle.load(open('mirror.pickle','rb'))
##############################
from termcolor import colored#
##############################
def word_preprocessing(x):
	white_space = ['..', '...', '/']
	none = ['.', ',', '"', '\n', '?', '!', '(', ')']
	x = [_removeNonAscii(s) for s in x]
	for _ in white_space: 
		x = [s.replace(_, ' ') for s in x]
	for _ in none: 
		x = [s.replace(_, '') for s in x]
	
	for sen in x:
		for word in sen.split():
			word_collection.add(word)
	'''
	for sen in x:
		if "ing" in sen:
			for word in sen.split():
				if "ing" in word:
					print(colored(word,'red'),end=' ')
				else:
					print(word , end=' ')
			print()
	'''
	final_x = []
	for sen in x:
		out = []
		for word in sen.split():
			if mirror[word] not in ['a','the','an','be']:
				if mirror[word] != 'hi':
					out.append(mirror[word])
				else:
					out.append('his')
				#print(mirror[word])		
		out = ' '.join(out)
		print(colored(sen,'cyan'))
		print(colored(out,'red'))
		final_x.append(out)
	return final_x


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
class sentence_repair:
	@staticmethod
	def trim_repeat(sentence):
		after = ['*HEAD*']
		for word in sentence.split():
			if word != after[-1]:
				after.append(word)
		return ' '.join(after[1:])
	@staticmethod
	def add_ing(word):
		if word.endswith('ie'):
			return word[:-2] + 'ing'
		if word.endswith('e'):
			return word[:-1] + 'ing'
		#son-mother-son
		if len(word) >=3 and word[-3] not in 'aeiou' and word[-2] in 'aeiou' and word[-1] not in 'aeiouy':
			return word + word[-1] + 'ing'
		return word+'ing'
	@staticmethod
	def verb_repair(sentence):
		# be-V repair
		sentence = sentence.split()
		for idx in range(len(sentence)):
			if sentence[idx] == 'be':
				if 'a' in sentence[max(0,idx-3) : idx] or \
					'the' in sentence[max(0,idx-3) : idx] :
					sentence[idx] = 'is'
				else:
					sentence[idx] = 'are'
		# be-V + Ving repair
		for idx in range(len(sentence)):
			if sentence[idx] in ['is' ,'are']:
				if idx+1 < len(sentence):
					#print(nltk.pos_tag([sentence[idx+1]]))
					if nltk.pos_tag([sentence[idx+1]])[0][1] in ['VB','VBZ' , 'VBD' , 'VBN' , 'VBG' , 'NN']:
						sentence[idx+1] = sentence_repair.add_ing(sentence[idx+1])
					
		return ' '.join(sentence)
	@staticmethod
	def plural(word):  
		if word.endswith('y'):  
			return word[:-1]+'ies'  
		elif word[-1] in 'sx' or word[-2:] in ['sh','ch']:  
			return word+'es'  
		elif word.endswith('an'):  
			return word[-2:]+'en'  
		else:  
			return word+'s'

	@staticmethod
	def can_be_verb(word):
		l = wn.synsets(word)
		counting = [0,0]
		for task in l:
			if task.pos() == 'v':
				counting[0] +=1
			else:
				counting[1] +=1
		#print(counting)
		if counting[0]*1.5 > counting[1] or word in ['slice']:
			#	print(word , 'is verb')
			return True
		return False			
	@staticmethod
	def simple_repair(sentence , no_trivial = True):
		sentence = sentence_repair.trim_repeat(sentence)
		sentence = sentence_repair.verb_repair(sentence)
		if no_trivial:
			# plural
			sentence = sentence.split()
			for idx in range(len(sentence)):
				word = sentence[idx]
				if nltk.pos_tag([word])[0][1] == 'CD' or word in ['some']:
					if idx + 1 < len(sentence):
						sentence[idx+1] = sentence_repair.plural(sentence[idx+1])
			# 'A' 'the' repair
			has_a = False
			for idx in range(len(sentence)):
				word = sentence[idx]
				if sentence_repair.can_be_verb(word)==False:
					if nltk.pos_tag([word])[0][1] == 'NN' and has_a == False:
						if word[0] not in 'aeiou':
							sentence[idx] = 'a ' + sentence[idx]
						else:
							sentence[idx] = 'an ' + sentence[idx]
						has_a = True
					elif nltk.pos_tag([word])[0][1] == 'NNS' and has_a == False:
						has_a = True
					elif nltk.pos_tag([word])[0][1].startswith('NN'):
						sentence[idx] = 'the ' + sentence[idx]
			
			# be-verb repair
			for idx in range(1,len(sentence)):
				word = sentence[idx]
				last_word = sentence[idx-1]
				if sentence_repair.can_be_verb(word)==True:
					if nltk.pos_tag([last_word])[0][1] == 'NN':
						sentence[idx] = 'is ' + sentence[idx]
					elif nltk.pos_tag([word])[0][1] == 'NNS':
						sentence[idx] = 'are ' + sentence[idx]
						
			sentence = ' '.join(sentence)
			sentence = sentence_repair.verb_repair(sentence)


			return sentence
			
		else:
			return sentence

if __name__ == '__main__':
	# print(sentence_repair.simple_repair("a beaa some trick play" , True))
	# print(sentence_repair.simple_repair("men ride women" , True))
	# print(sentence_repair.can_be_verb('play'))
	# print(sentence_repair.can_be_verb('man'))
	# print(sentence_repair.can_be_verb('cook'))
	# print(sentence_repair.can_be_verb('ride'))
	print(sentence_repair.simple_repair("boy play" , True))

	print('NEXT')
class caption_test:
	def __init__(self):
		file_lists = [ filename for filename in os.listdir(test_path)]
		self.file_lists = file_lists
		self.test_data = [np.load(test_path + '/' + filename) for filename in file_lists]
		self.N = len(file_lists)
		self.next_batch_counter = 0
		print('test_data initization finished.')
	def next_batch(self,batch_size):
		if self.next_batch_counter + batch_size < self.N:
			output = self.test_data[self.next_batch_counter:self.next_batch_counter+batch_size]
			self.next_batch_counter += batch_size
			if self.next_batch_counter == self.N:
				self.next_batch_counter = 0
		else:
			output = []
			for i in range(batch_size):
				output.append(self.test_data[self.next_batch_counter])
				self.next_batch_counter += 1
				if self.next_batch_counter >=self.N:
					self.next_batch_counter = 0
		return output
	
	def single_test(self):
		feat_name = self.file_lists[self.next_batch_counter]
		feat = self.test_data[self.next_batch_counter]
		self.next_batch_counter += 1
		if self.next_batch_counter >= self.N:
			self.next_batch_counter = 0
		return feat , feat_name

class caption_data:

	def __init__(self , word_min_frequency=3 , verbose = True):
		# read file
		with open(label_path,'r') as fin:
			self.allData = pd.DataFrame(json.load(fin))

		# word preprocessing
		self.N = self.allData.shape[0]
		self.allData['caption'] = self.allData['caption'].apply(word_preprocessing)

		# build dictionary
		word_counter = np.zeros((20000))
		self.word_dim = 0	
		self.D = {}

		## add special character

		for voc in ['<bos>' , '<eos>' , PAD , '<unk>']:
			self.D.update({voc:self.word_dim})
			word_counter[self.D[voc]] = 99999
			self.word_dim += 1

		## add vocab in dataframe
		def build_dictionary(x):
			for s in x:
				for voc in s.split():
					if voc not in self.D:
						self.D.update({voc:self.word_dim})
						self.word_dim += 1
					word_counter[self.D[voc]] += 1
		self.allData['caption'].apply(build_dictionary)

		# filter word_dim by frequency
		self.D = {i:self.D[i] for i in self.D if word_counter[self.D[i]] >= word_min_frequency}
		self.word_dim = len(self.D)
		self.D = {i:j for i, j in zip(self.D, [_ for _ in range(self.word_dim)])}
		self.inv_D = { self.D[key] : key for key in self.D}

		# Data Cleaning
		self.sen_max_length = 25
		def data_cleaning(x):
			cleaning = []
			for sen in x:
				new_sen = ['<bos>']
				for voc in sen.split():
					if voc not in self.D:
						new_sen.append('<unk>')
					else:
						new_sen.append(voc)
				s = " ".join(new_sen)
				cleaning.append(s)
			return cleaning

		self.allData['caption'] = self.allData['caption'].apply(data_cleaning)
		print('(data preprocessing) quantity of videos: ',self.N)
		print('(data preprocessing) quantity of vocs  : ',self.word_dim)
		print ("max_length = ", self.sen_max_length)
		## FOR OTHER METHOD TO USE ##
		self.next_batch_counter = 0
		self.inv_D = { self.D[key] : key for key in self.D}
		
	def getAFullCap(self,idx):
		filename = self.allData.loc[idx, 'id']
		print(self.allData.loc[idx, 'caption'])
		return np.load(feat_path + '/'+ filename+'.npy')


	def getASingleCap(self,idx):
		cur_len = 0
		while cur_len == 0 or cur_len > self.sen_max_length:
			import random
			out_idx = [ i for i in range(len(self.allData.loc[idx, 'caption']))]
			for each_idx in range(len(self.allData.loc[idx,'caption'])):
				each_sen = self.allData.loc[idx,'caption'][each_idx]
				if 'and' in each_sen.split():
					# if sentence contained 'and', the prob *=5 
					out_idx += [each_idx] * 5
			random.shuffle(out_idx)
			#print(out_idx)
			sen_idx = out_idx[0]
			
			sen_idx = np.random.randint(0,len(self.allData.loc[idx, 'caption']))
			now_sen_list = self.allData.loc[idx, 'caption'][sen_idx].split()
			cur_len = len(now_sen_list)

		output = []
		for i,voc in zip(range(len(now_sen_list)),now_sen_list):
			try:
				output.append(self.D[voc])
			except:
				output.append(self.D['<unk>'])
		filename = self.allData.loc[idx, 'id']
		#print(now_sen)
		return np.load(feat_path + '/'+ filename+'.npy'),output

	def getFiveCap(self,idx):
		import random
		out_idx = [ i for i in range(len(self.allData.loc[idx, 'caption']))]
		random.shuffle(out_idx)
		out_idx = out_idx[:5]
		output = []
		for sen_idx in out_idx:
			print(self.allData.loc[idx, 'caption'][sen_idx])
			now_sen_list = self.allData.loc[idx, 'caption'][sen_idx].split()
			cur_len = len(now_sen_list)

			now_output = []
			for i,voc in zip(range(len(now_sen_list)),now_sen_list):
				try:
					now_output.append(self.D[voc])
				except:
					now_output.append(self.D['<unk>'])
			output.append(now_output)
		filename = self.allData.loc[idx, 'id']
		#print(now_sen)
		return np.load(feat_path + '/'+ filename+'.npy'),output

	def next_batch(self,batch_size):
		output = []
		feat_output = []
		max_length = 0
		for _ in range(batch_size):
			if self.next_batch_counter >= self.N:
				self.next_batch_counter = 0
			#print('N:' , self.N , self.next_batch_counter)
			feat,sing = self.getASingleCap(self.next_batch_counter)
			max_length = max(max_length , len(sing))
			output.append(sing)
			feat_output.append(feat)
			self.next_batch_counter += 1
		
		for sing in output:
			sing.append(self.D['<eos>'])
			sing += [ self.D[PAD] for _ in range(max_length-len(sing)+1) ]
		#exit()
		return np.array(feat_output),np.array(output)
	
	def next_batch_with_same(self,batch_size):
		output = []
		feat_output = []
		max_length = 0
		for _ in range(batch_size):
			if self.next_batch_counter >= self.N:
				self.next_batch_counter = 0
			#print('N:' , self.N , self.next_batch_counter)
			feat,sing = self.getFiveCap(self.next_batch_counter)
			max_length = max(max_length , max( [len(s) for s in sing]))
			output += sing
			feat_output += [feat] *5
			self.next_batch_counter += 1
		
		for sing in output:
			sing.append(self.D['<eos>'])
			sing += [ self.D[PAD] for _ in range(max_length-len(sing)+1) ]
		#exit()
		return np.array(feat_output),np.array(output)

	def one_to_sen(self,one_hot):
		return  ' '.join([ self.inv_D[idx] for idx in one_hot])

	def predict(self, one_hot):
		word = []
		for idx in one_hot[1:]:
			if self.inv_D[idx] == PAD:
				pass
			elif self.inv_D[idx] == '<eos>':
				break
			else:
				word.append(self.inv_D[idx])
		s = sentence_repair.simple_repair(' '.join(word)) + '.'
		return s.capitalize()

if __name__ == '__main__':
	# print(sentence_repair.trim_repeat("cat a a a a cat be be be sth"))
	# print(sentence_repair.verb_repair("cat a a a a cat be be be sth"))
	# print(sentence_repair.verb_repair("a cat be cook"))
	# print(sentence_repair.simple_repair("a monkey be on a a "))
	# print(sentence_repair.simple_repair("a be be peel shrimp"))
	# print(sentence_repair.simple_repair("a be"))
	#print(sentence_repair.simple_repair("a be" , True))
	print()

	print()



	C = caption_data(verbose = False)
	C.getASingleCap(790)
	feat , output = C.getFiveCap(790)
	print(output)
	get = C.next_batch_with_same(5)[0]
	print(get[0])
	print(get[1])
	print(get[6])

	print(C.next_batch_with_same(5)[0].shape)

	print(C.next_batch_with_same(5)[1].shape)
	#import numpy as np
	#np.save('word_collection',word_collection)
	#print(word_collection)
	'''
	C.getAFullCap(790)
	print(C.getASingleCap(877)[1])
	for row in C.next_batch(10)[1]:
		print(row)
	'''
	#CC = caption_test()

	#print(CC.next_batch(50))
#for i in C.allData:
#	print(len(i['caption']))
#print(C.next_batch(10))
#print(C.next_batch(10))

