import re
import json
import time
import cv2
import  threading
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from skimage import io
from skimage import transform

hair_dict = {
		'orange': 0, 	'white': 1, 	'aqua': 2, 
		'grey': 3, 		'green': 4, 	'red': 5, 
		'purple': 6, 	'pink': 7,		'blue': 8, 
		'black': 9, 	'brown': 10, 	'blonde': 11,	'gray': 12
	}
eyes_dict = {
		'black': 0, 	'orange': 1, 	'pink': 2, 
		'yellow': 3, 	'aqua': 4, 		'purple': 5, 
		'green': 6, 	'brown': 7,		'red': 8, 		'blue': 9
	}

hair_inv_dict = { v:k for k,v in hair_dict.items()}
eyes_inv_dict = { v:k for k,v in eyes_dict.items()}
class Danager:
	def __init__(self,batch_size , img_path= None , tag_path = None , workers = 1 ,verbose=True, minal= 0 , maxal = 200001):
		if img_path == None:
			img_path = '../data/extra_extra_data/face/'
			self.img_path = img_path
		if tag_path == None:
			tag_path = '../data/extra_extra_data/'
			self.tag_path = tag_path
		########## param ##########
		self.margin = 28
		self.img_size = 128
		self.max_pool_size= 20
		###########################
		self.verbose = verbose
		self.tag_dict = {}
		self.batch_pool = deque()
		print('tag is merging.')
		self.merge_tag()
		print('tag is prepared.')
		def append_batch(self , th ):
			while True:
				this_batch_img = []
				this_batch_tag = []
				while len(this_batch_img) < batch_size:
					try:
						idx = np.random.randint(minal,maxal)
						this_img = self.read_img(idx)
						this_tag_text = self.tag_dict[idx]
						hairColor , _ , eyesColor , _ = this_tag_text.split()
						this_tag = np.zeros(23)
						this_tag[hair_dict[hairColor]] = 1
						this_tag[13 + eyes_dict[eyesColor]] = 1

						this_batch_img.append(this_img)
						this_batch_tag.append(this_tag)
						# print('OK!')
					except:
						# print('Not OK, QQ!')
						pass
				this_batch_img = np.array(this_batch_img)
				this_batch_tag = np.array(this_batch_tag)

				while len(self.batch_pool) >= self.max_pool_size:
					time.sleep(1)

				self.batch_pool.append((this_batch_img , this_batch_tag))
				
		for i in range(workers):
			now_thread = threading.Thread(target = append_batch , args=(self,i))  
			now_thread.start()
		self.all_waiting_time = 0
		self.get_times=1

	def get_batch(self):
		time_wait = 0
		while len(self.batch_pool) == 0:
			if self.verbose:
				print('deque is empty, wait a second, your waiting time: {:.1f} , avg waiting time: {:.2f}'.format(time_wait , self.all_waiting_time/self.get_times) , end='\r')

			time_wait += 0.1
			self.all_waiting_time += 0.1
			time.sleep(0.1)
		if time_wait !=0 :
			if self.verbose:
				print()
		self.get_times +=1
		return self.batch_pool.popleft()
	
	def read_img(self,idx):
		try:
			img_f = self.img_path + '/' + str(idx) + '.jpg'
			img = np.array(io.imread(img_f)).astype(np.float64)
			# print(img)
			margin = self.margin
			img = img[margin:-margin,margin:-margin]
			img = cv2.resize(img,(self.img_size,self.img_size))#,interpolation=cv2.INTER_CUBIC')
			return (img *2 /255.) - 1
		except :
			return None
	def merge_tag(self):
		all_data = []
		for file_idx in range(5):
			with open(self.tag_path + '/tags' + str(file_idx) + '.txt' , 'r') as fin:
				data = fin.read()
			all_data += data.split('\n')
		for row in all_data:
			try:
				tag_idx , this_tag = row.split(',')
				self.tag_dict.update({int(tag_idx) : this_tag})
			except:
				pass
	@staticmethod
	def regular_result( gen_imgs ,tag = None, save_file = None  , r=4,c=6):
		gen_imgs = np.array(gen_imgs)
		gen_imgs = ( (gen_imgs+1)/2 * 255).astype(np.uint8)
		fig, axs = plt.subplots(r, c)
		cnt = 0
		for i in range(r):
			for j in range(c):
				extra_margin = 2
				if extra_margin is None:
					this_img = cv2.resize(gen_imgs[cnt],(64,64))
				else:
					this_img = cv2.resize(gen_imgs[cnt][extra_margin:-extra_margin,extra_margin:-extra_margin],(64,64))
				axs[i,j].imshow(this_img)
				axs[i,j].axis('off')
				if tag is not None:
					t = tag[cnt]
					hr = np.argmax(t[:13])
					ey = np.argmax(t[13:])
					axs[i,j].set_title(	\
						hair_inv_dict[hr]+'/'+eyes_inv_dict[ey]
						,fontdict={'fontsize':8}
					)
				cnt += 1

		if save_file is not None:
			fig.savefig(save_file)
		else:
			plt.show()
		plt.close()


if __name__ == '__main__':		
	danager = Danager(32,workers=1)
	while True:
		i , t = danager.get_batch()
		danager.regular_result(i,t)

	danager.regular_result(i , t )
	print(i.shape)
	print(t.shape)
	'''
	OAO = []
	for i in range(200000,199974,-1):
		get = danager.read_img(i)
		OAO.append(get)
	danager.regular_result(np.array(OAO))
	'''
	'''
	while True:
		print('get',danager.get_batch())
	'''
