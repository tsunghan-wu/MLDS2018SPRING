import numpy as np
import pandas as pd
import json
import pickle
import os
import sys
#test_path  = 'MLDS_hw2_1_data/testing_data/feat'
test_path = sys.argv[1] + '/feat' 
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
if __name__ == '__main__':
	import pickle
	
	test_data = caption_test()
	pickle.dump(test_data,open('reduce_test_data' ,'wb'))
	