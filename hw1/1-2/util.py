import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
<<<<<<< HEAD
=======

def FC_layer(input_shape,neuron,inc , dev=0.1):
	W = tf.Variable( tf.truncated_normal([input_shape,neuron], stddev=dev))
	b = tf.Variable( tf.truncated_normal([neuron], stddev=dev))
	return tf.matmul(inc,W) + b
>>>>>>> 910b511e78ff7ede38926da85ab887eb42402a3b
class seq:

	def __init__( self , x , y ,input_shape):
		self.now_param = [input_shape]
		self.x  , self.now_output = x , x
		self.y , self.input_shape = y , input_shape
<<<<<<< HEAD
		self.tracable = None
		self.tracablev2 = []
	'''
	def create_variables(self,shape , dev):
		now_params = np.prod(shape)
		flat = tf.Variable( tf.truncated_normal([now_params], stddev=dev))
		prototype = flat
		if self.tracable is None:
			self.tracable = flat
			prototype = tf.reshape(flat,shape)
		else : 
			self.tracable = tf.concat( [flat,self.tracable] , axis = 0 )
			flat2 = tf.split(self.tracable , [now_params,-1] )[0]
			print(tf.split(self.tracable , [now_params,-1] ))
			print(self.tracable)
			prototype = tf.reshape(flat2,shape)

		return prototype 
	'''
	def create_variables(self,shape , dev):
		now_params = 1
		for dim in shape:
			now_params *= dim
		print(now_params , shape)
		flat = tf.Variable( tf.truncated_normal([now_params], stddev=dev))
		sp = np.zeros(now_params,dtype=np.int32) + 1 
		if now_params != 1:
			sp[-1 ] = -1
		all_shattered = tf.split(flat,sp)
		self.tracablev2 += all_shattered
		flat = tf.concat(all_shattered , axis=0)
		prototype = tf.reshape(flat,shape)
		return prototype 
	
	def FC_layer(self,input_shape,neuron,inc , dev=0.1):
		W =self.create_variables([input_shape,neuron], dev=dev)
		b =self.create_variables([neuron], dev=dev)
		return tf.matmul(inc,W) + b

	def add_FC(self,size,stddev=0.1):
		self.now_output = self.FC_layer(self.input_shape,size,self.now_output,0.1)
=======
		

	def add_FC(self,size):
		self.now_output = FC_layer(self.input_shape,size,self.now_output)
>>>>>>> 910b511e78ff7ede38926da85ab887eb42402a3b
		self.input_shape = size
		self.now_param.append(self.input_shape)
	
	def add_activate(self , activator):
		self.now_output = activator(self.now_output)
<<<<<<< HEAD
	
	def get_grad_norm_init(self):
		grads = self.optimizer.compute_gradients(self.loss)
		grads, _ = list(zip(*grads))
		flat = []
		for var in grads:
			flat.append(tf.reshape(var,[-1]))
		self.grad_norm2 = tf.norm(tf.concat(flat,axis=0) , 2)
		return self.grad_norm2

	def get_loss_hesssians_init(self):
		table = []
		print('compile hessians....')
		for v1 in self.tracablev2:
			row = []
			for v2 in self.tracablev2:
				row.append(tf.gradients(tf.gradients(self.loss, v2)[0], v1)[0])
			row = tf.stack([t for t in 	row])
			table.append(row)
		table = tf.stack(table)
		self.loss_hessians = table
		print('compile hessians finished....')
		return  self.loss_hessians

	def eigen(self):
		tf.self_adjoint_eig()

	def get_grad_hessians(self , x , y):
		return self.sess.run(self.loss_hessians,feed_dict={
				self.x : x,
				self.y : y
			})


	def get_grad_norm(self , x , y):
		return self.sess.run(self.grad_norm2,feed_dict={
				self.x : x,
				self.y : y
			})
	
	def get_train(self , sess_ , optimizer = tf.train.AdamOptimizer(0.01) ,needs_Hessians = False):
		### normal train step ###
		self.sess = sess_
		self.loss = tf.reduce_mean(tf.square( self.now_output - self.y ))
		self.optimizer = optimizer
		self.train_step = self.optimizer.minimize(self.loss)
		### some initialization ###
		self.trainable = tf.trainable_variables()
		self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.now_output, 1)) 
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		self.get_grad_norm_init()
		if needs_Hessians:
			self.get_loss_hesssians_init()
		
		return self.now_output,self.train_step

	def get_zero_norm_train(self  , optimizer = tf.train.AdamOptimizer(0.01)):
		self.zero_norm_train_step = optimizer.minimize(self.get_grad_norm_init())
		return self.zero_norm_train_step
		
=======

	def get_train(self , sess_):
		self.sess = sess_
		# self.loss = tf.reduce_mean(tf.square( self.now_output - self.y ))
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.now_output))
		self.optimizer = tf.train.AdamOptimizer(0.01)
		#self.optimizer = tf.train.GradientDescentOptimizer(0.5)
		self.train_step = self.optimizer.minimize(self.loss)
		self.correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.now_output, 1)) 
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		return self.now_output,self.train_step

>>>>>>> 910b511e78ff7ede38926da85ab887eb42402a3b
	def get_loss(self , x , y):
		return self.sess.run(self.loss,feed_dict={
				self.x : x,
				self.y : y
			})
	def get_acc(self , x , y):
		return self.sess.run(self.accuracy,feed_dict={
				self.x : x,
				self.y : y
			})	
	def save_model(self,path):
		saver = tf.train.Saver()
		saver.save(self.sess, path + "/model.ckpt")

	def load_model(self,path):
		saver = tf.train.Saver()
		saver.restore(self.sess, path + "/model.ckpt")
<<<<<<< HEAD
	
	def get_whole_variable(self):
=======
	def save_whole_variable(self,path):
>>>>>>> 910b511e78ff7ede38926da85ab887eb42402a3b
		param = np.empty(shape=[0, 1])
		variables_names = [v.name for v in tf.trainable_variables()]
		values = self.sess.run(variables_names)
		for k, v in zip(variables_names, values):
			param = np.append(param, v.reshape(-1, 1)).reshape(-1, 1)
<<<<<<< HEAD
		return param

	def save_whole_variable(self,path):
		param  = self.get_whole_variable()
		F = open(path, 'a')
		np.savetxt(F, param)
		F.close()
		
=======
		F = open(path, 'a')
		np.savetxt(F, param)
		F.close()
>>>>>>> 910b511e78ff7ede38926da85ab887eb42402a3b
	def save_one_layer(self,path):
		param = np.empty(shape=[0, 1])
		variables_names = ["Variable:0", "Variable_1:0"]
		values = self.sess.run(variables_names)
		for k, v in zip(variables_names, values):
			param = np.append(param, v.reshape(-1, 1)).reshape(-1, 1)
		F = open(path, 'a')
		np.savetxt(F, param)
		F.close()
			
	def summary(self):
		all_param = 0
		for _ in range(len(self.now_param)-1):
			all_param+= self.now_param[_] * self.now_param[_+1] + self.now_param[_+1]
		return all_param


class gif_gen:
	def __init__(self , left, right , interval , ansx , ansy):
		self.fig, self.ax = plt.subplots()
		self.fig.set_tight_layout(True)
		self.storage = []
		self.ans_line, = self.ax.plot(ansx, ansy, 'r-', linewidth=2 , color='red')
		self.pred_line, = self.ax.plot(ansx, ansy, 'r-', linewidth=2,color='blue')

		print('fig size: {0} DPI, size in inches {1}'.format(
			self.fig.get_dpi(), self.fig.get_size_inches()))

	def set_frame(self,x,y):
		self.storage.append( (x,y) )

	def update(self, i , label='epoch {0}'):
		label = label.format(i*500)
<<<<<<< HEAD
=======
		
		# 更新直线和x轴（用一个新的x轴的标签）。
		# 用元组（Tuple）的形式返回在这一帧要被重新绘图的物体
>>>>>>> 910b511e78ff7ede38926da85ab887eb42402a3b
		self.pred_line.set_xdata(self.storage[i][0])
		self.pred_line.set_ydata(self.storage[i][1])
		self.ax.set_xlabel(label)
		return self.pred_line, self.ax

	def output(self ,path='line.gif' , inter = 80):
		anim = FuncAnimation(self.fig, self.update, 
			frames=np.arange(0, len(self.storage)), interval=inter)
		anim.save(path, dpi=80, writer='imagemagick')
		plt.show()