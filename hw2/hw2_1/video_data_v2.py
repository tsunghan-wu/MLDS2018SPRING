import os
import json
import nltk
import pickle
import random

from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

import numpy as np
import pandas as pd

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

#label_path = 'MLDS_hw2_1_data/training_label.json'
label_path = 'training_label.json'
feat_path  = 'MLDS_hw2_1_data/training_data/feat'


caption = json.load(open(label_path,'r'))

OC_list = []
early = 0
def add_ing(word):
	# Exception list
	if word in ['seasoning']:
		return word
	if word is "se":
		return "seeing"
	if word is 'fle':
		return 'fleeing'
	if word in ['mix' , 'rid' , 'encounter','powder' , 'hammer' ,  'fix',
				'open' , 'sharpen' , 'draw' , 'season' , 'deliver' , 'gallop' , 
				'straighten' , 'chew' , 'chisel', 'split' , 'throw' , 'slow' , 
				'blow' , 'sew' , 'discover' , 'darken' ,'show']:
		return word+'ing'
	if word.endswith('ie'):
		return word[:-2] + 'ing'
	if word.endswith('e'):
		return word[:-1] + 'ing'
	#son-mother-son
	if len(word) >=3 and word[-3] not in 'aeiou' and word[-2] in 'aeiou' and word[-1] not in 'aeiouy':
		return word + word[-1] + 'ing'
	return word+'ing'

def is_noun(analyze):
	return analyze[1] in ['NN' , 'NNS'] or analyze[0] in [
		'puppy' , 'can' , 'animal' ,'human'
	]
def sentence_spliting(sentence , cutting = True):

	def _removeNonAscii(s):
		return "".join([i for i in s if ord(i)<128])

	next_sentence = []
	for word in sentence.lower().split():
		if word.endswith("'s"):
			word = word[:-2] + ' ' + "'s"
		next_sentence.append(word)
	sentence = ' '.join(next_sentence)
	white_space = ['..', '...', '/']
	none = ['.', ',', '"', '\n', '?', '!', '(', ')']
	x = [_removeNonAscii(s) for s in sentence]
	for _ in white_space: 
		x = [s.replace(_, ' ') for s in x]
	for _ in none: 
		x = [s.replace(_, '') for s in x]
	sentence = ''.join(x)

	if not ord('a') < ord(sentence[-1]) < ord('z'):
		sentence = sentence[:-1]
	sentence = sentence.lower().split()
	for idx in range(len(sentence)):
		if sentence[idx] in ['his']:
		#if sentence[idx] in ['the' , 'his']:
			if sentence[idx+1][0] in 'aeiou':
				sentence[idx] = 'an'
			else:
				sentence[idx] = 'a'
	next_sentence = []
	for word in sentence:
		if word not in ['some']:
			next_sentence.append(word)
	sentence = next_sentence


	fixed_dict={
		'violen' 	: 'violin',
		'voilin' 	: 'violin',
		'wa7ter' 	: 'water',
		'violinist'	: 'violin player',
		'vending' 	: 'vend',
		'vegetables': 'vegetable',
		'vegetation': 'vegetable',
		'vegatable' : 'vegetable',
		'vanill'	: 'vanilla',
		'urinates'	: 'urinate',
		'unwraps'	: 'unwrap',
		'umbrell'	: 'umbrella',
		'tun'		: 'tuna',
		'traditonal': 'traditional',
		'townspeople':'people in town',
		'toilette'	: 'toilet',
		'thump'		: 'hit',
		'tempur'	: 'tempura',
		'tortill'	: 'tortilla',
		'teenaged'	: 'teenage',
		'te'		: 'tea',
		'streetcar' : 'street car',
		'stire'		: 'stir',
		'stics'		: 'stick',
		'squirrrel'	: 'squirrel',
		'squid'		: 'squirrel',
		'squirm'	: 'squirrrel',
		'soccar'	: 'soccer',
		'sof'		: 'soft',
	}

	next_sentence = []
	for word in sentence:
		if word in fixed_dict:
			word = fixed_dict[word]
		next_sentence.append(word)
	sentence = next_sentence


	if len(sentence) >= 15:
		return ''
	if not cutting :
		return sentence
	next_sentence = []
	for word_ana in nltk.pos_tag(sentence):
		word = word_ana[0]
		if 	word_ana[1] == 'VBZ' and \
			word_ana[0] not in ['is' , "'s" , 'languages' , 
				'empties' , 'eyes' ,'noodles' , 'vegetables' , 
				'coats' ,'guitars' , 'boys']:
			word = "is " + add_ing(lemmatizer.lemmatize(word_ana[0],pos='v'))
		next_sentence.append(word)
	sentence = next_sentence	
	analyze = nltk.pos_tag(sentence)
	
	delete_list = []
	for word in analyze:
		if word[0] not in [
			'trashcan' , 'oven', 'other',
			'down','up' , 'mixed' ,'puppy' , 
			'veterinarian' , 'jack-o-lantern'] and nltk.pos_tag([word[0]])[0][1] in ['JJ' , 'RB']:
			delete_list.append(word[0])
	
	if delete_list:
		sentence = []
		
		for word in analyze:
			if word[0] not in delete_list:
				#print(word[0] , end=' ')
				sentence.append(word[0])
		#print('\t delete:' , nltk.pos_tag(delete_list))
	
	# a ginger -> ginger
	# some ginger -> ginger
	next_sentence = []
	for idx,word in zip(range(40),sentence):
		if not word:
			continue
		if nltk.pos_tag([word])[0][1] in ['DT' , 'CD']:
			if idx+2 < len(sentence) and \
			nltk.pos_tag([sentence[idx+2]])[0][1] in ['NN','NNS'] :
			#and (sentence[idx+1].endswith('ing') or sentence[idx+1].endswith('ed')):
				next_sentence.append(word + ' ' + sentence[idx+1]+ ' ' + sentence[idx+2])
				sentence[idx+1] = ""
				sentence[idx+2] = ""
			elif idx+1 < len(sentence) and nltk.pos_tag([sentence[idx+1]])[0][1] in ['NN','NNS']:
				next_sentence.append(word + ' ' + sentence[idx+1])
				sentence[idx+1] = ""
			else:
				next_sentence.append(word)
		else:
			next_sentence.append(word)

	if len(next_sentence) < 3:
		return ''
	
	sentence = next_sentence
	next_sentence = []
	for idx,word in zip(range(40) , sentence):
		if not word:
			continue
		if word in ['is' , 'are']:
			for iidx in range(idx+1,len(sentence)):
				if sentence[iidx].endswith('ing'):
					for iiidx in range(idx+1,iidx+1):
						word += ' ' + sentence[iiidx]
						sentence[iiidx] = ""
					break
			
		next_sentence.append(word)
	
	sentence = next_sentence

	next_sentence = []
	for idx in range(len(sentence)-1):
		if sentence[idx] == '':
			continue
		if len(sentence[idx].split()) == 1 and len(sentence[idx+1].split()) == 1  and\
		nltk.pos_tag([sentence[idx]])[0][1] in ['NN' , 'NNS'] and nltk.pos_tag([sentence[idx+1]])[0][1] in ['NN' , 'NNS']:	
			sentence[idx] = sentence[idx] + ' '+ sentence[idx+1]
			sentence[idx + 1 ] = ''
		next_sentence.append(sentence[idx])
	next_sentence.append(sentence[-1])
		

	if len(next_sentence) <= 1 :
		return ''

	sentence = [ word for word in next_sentence if word != '']
	return sentence


class caption_data:
	
	def __init__(self,min_frequency = 3):
		counting = []
		ite = 0
		self.captions = []
		self.filenames = []
		self.count_voc = {}
		for data in caption:
			now_caption = []

			self.filenames.append(data['id'])
			for sentence in data['caption']:
				ite +=1
				# if ite > 100:
				# 	break
				split_res = sentence_spliting(sentence)
				if not split_res:
					continue
				for word in split_res:
					if word not in self.count_voc:
						self.count_voc.update({word:0})
					self.count_voc.update({word:self.count_voc[word]+1})
				counting += split_res
				org_res = sentence_spliting(sentence,False)
				if not org_res:
					continue
				for word in org_res:
					if word not in self.count_voc:
						self.count_voc.update({word:0})
					self.count_voc.update({word:self.count_voc[word]+1})
				counting += org_res
				
				#print(sentence , '->' ,sentence_spliting(sentence))
				now_caption.append(split_res)
			self.captions.append(now_caption)
		'''
		for item in self.captions:
			if item:
				print(item)
		'''
		self.D = {'<bos>' : 0,'<eos>':1,'<pad>':2,'<unk>':3}
		self.inv_D = { 0: '<bos>' ,1: '<eos>',2: '<pad>',3: '<unk>'}
		now_idx = 4
		for key in self.count_voc:
			if self.count_voc[key] >= min_frequency:
				print(key , self.count_voc[key])
				self.D.update({key : now_idx})
				self.inv_D.update({now_idx : key})
				now_idx +=1
			else:
				self.D.update({key : self.D['<unk>']})
		self.word_dim = len(self.D)
		print('\nword_dim:' , self.word_dim)
		self.batch_counter = 0

		next_captions = []
		for data in self.captions:
			this_caption = []
			for sentence in data:
				next_sentence = []
				for word in sentence:
					if self.D[word] == self.D['<unk>']:
						# v-ing -> is v-ing
						def has_dict(word):
							return word in self.D and self.D[word] != self.D['<unk>']

						if has_dict('is '+word):
							next_sentence.append('is '+ word)
						elif (word + 's') in self.D and self.D[word+ 's'] != self.D['<unk>']:
							next_sentence.append(word+ 's')
						elif len(word.split()) == 3:
							tmp = word.split()
							if has_dict(tmp[0] + ' '+ tmp[2]):
								next_sentence.append(tmp[0] + ' '+ tmp[2])
							elif has_dict(tmp[1] + ' '+ tmp[2]):
								next_sentence.append(tmp[1] + ' '+ tmp[2])
							elif has_dict(tmp[0] + ' '+ tmp[1]):
								next_sentence.append(tmp[0] + ' '+ tmp[1])

						else:	
							pass
							#next_sentence.append(word)
							for splited_word in word.split():
								next_sentence.append(splited_word)
					else:
						next_sentence.append(word)
				if '<unk>' in self.one_to_sen(self.sentence_2one_hot(sentence)).split():
					print('|'.join(sentence))
					print(self.one_to_sen(self.sentence_2one_hot(sentence)))
					print('|'.join(next_sentence))
					print(nltk.pos_tag(' '.join(next_sentence).split()))
					print()

				
				this_caption.append(next_sentence)
			next_captions.append(this_caption)
		self.captions = next_captions
		for data in self.captions:
			for sentence in data:
				print('|'.join(sentence))
				print(self.one_to_sen(self.sentence_2one_hot(sentence)) , end='\n\n')
	def sentence_2one_hot(self,sentence):
		output = []
		for word in sentence:
			if word in self.D:
				output.append(self.D[word])
			else:
				output.append(self.D['<unk>'])
		return [self.D['<bos>']] + output + [self.D['<eos>']]
	def one_to_sen(self,sentence):
		return ' '.join([ self.inv_D[word] if word in self.inv_D else '<no_exist>' for word in sentence])
	def get_cap(self,idx):
		caption_length = len(self.captions[idx])
		sentence_idx = random.randint(0,caption_length-1)		
		'''
		print(self.captions[idx][sentence_idx])
		print(self.sentence_2one_hot(self.captions[idx][sentence_idx]))
		print(self.one_to_sen(self.sentence_2one_hot(self.captions[idx][sentence_idx])))
		print(self.filenames[idx])
		'''

		return np.load(feat_path + '/'+ self.filenames[idx]+'.npy') , self.sentence_2one_hot(self.captions[idx][sentence_idx])

	def	next_batch(self,batch_size):
		output=[[],[]]
		# feat , caption
		max_length = 0
		for _ in range(batch_size):
			res = self.get_cap(self.batch_counter)
			output[0].append(res[0])
			output[1].append(res[1])
			max_length = max(max_length,len(res[1]))
			self.batch_counter += 1
			if self.batch_counter == len(self.captions):
				self.batch_counter= 0
		# padding
		for feat in output[1]:
			feat += [self.D['<pad>']] * (max_length - len(feat)) 
		#print('max_length',max_length)
		return (np.array(output[0]) , np.array(output[1]))
	def predict(self,sentence):
		sentence = self.one_to_sen(sentence).split()
		next_sentence = []
		for word in sentence:
			if word not in ['<bos>' , '<eos>' , '<pad>' , '<unk>']:
				next_sentence.append(word)


		return ' '.join(next_sentence)
if __name__ == '__main__':
	data = caption_data(4)
	pickle.dump(data , open('processed_training_data_v2','wb'))
