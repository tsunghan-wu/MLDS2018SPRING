import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
class seq:

	def __init__( self , x , y ,input_shape):
		self.now_param = [input_shape]
		self.x  , self.now_output = x , x
		self.y , self.input_shape = y , input_shape
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
			prototype = tf.reshape(flat2,shape)

		return prototype 
	'''
	def create_variables(self,shape , dev):
		now_params = np.prod(shape)
		flat = tf.Variable( tf.truncated_normal([now_params], stddev=dev))
		sp = np.zeros(now_params,dtype=np.int32)  + 1 
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
		self.input_shape = size
		self.now_param.append(self.input_shape)
	
	def add_activate(self , activator):
		self.now_output = activator(self.now_output)
	
	def get_grad_norm_init(self):
		grads = self.optimizer.compute_gradients(self.loss)
		grads, _ = list(zip(*grads))
		flat = []
		for var in grads:
			flat.append(tf.reshape(var,[-1]))
		self.grad_norm2 = tf.norm(tf.concat(flat,axis=0) , 2)
		return self.grad_norm2

	def get_loss_hesssians_init(self):
		#print(self.tracablev2)
		print('h start')
		self.loss_hessians = tf.hessians(xs = self.tracablev2 , ys=self.loss)
		print('h end')
		return  self.loss_hessians

	def eigen(self):
		tf.self_adjoint_eig()

	def get_grad_hessians(self , x , y):
		return self.sess.run(self.second_order,feed_dict={
				self.x : x,
				self.y : y
			})


	def get_grad_norm(self , x , y):
		return self.sess.run(self.grad_norm2,feed_dict={
				self.x : x,
				self.y : y
			})
	
	def get_train(self , sess_ , optimizer = tf.train.AdamOptimizer(0.01)):
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
		self.get_loss_hesssians_init()
		
		return self.now_output,self.train_step

	def get_zero_norm_train(self  , optimizer = tf.train.AdamOptimizer(0.01)):
		self.zero_norm_train_step = optimizer.minimize(self.get_grad_norm_init())
		return self.zero_norm_train_step
		
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
	
	def get_whole_variable(self):
		param = np.empty(shape=[0, 1])
		variables_names = [v.name for v in tf.trainable_variables()]
		values = self.sess.run(variables_names)
		for k, v in zip(variables_names, values):
			param = np.append(param, v.reshape(-1, 1)).reshape(-1, 1)
		return param

	def save_whole_variable(self,path):
		param  = self.get_whole_variable()
		F = open(path, 'a')
		np.savetxt(F, param)
		F.close()
		
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
		self.pred_line.set_xdata(self.storage[i][0])
		self.pred_line.set_ydata(self.storage[i][1])
		self.ax.set_xlabel(label)
		return self.pred_line, self.ax

	def output(self ,path='line.gif' , inter = 80):
		anim = FuncAnimation(self.fig, self.update, 
			frames=np.arange(0, len(self.storage)), interval=inter)
		anim.save(path, dpi=80, writer='imagemagick')
		plt.show()