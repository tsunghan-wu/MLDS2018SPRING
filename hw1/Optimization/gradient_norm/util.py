### for MLDS HW 1-2-2, 1-2-3 (gradient norm + gradoent zero) ###
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def FC_layer(input_shape,neuron,inc , dev=1):
	W = tf.Variable( tf.truncated_normal([input_shape,neuron], stddev=dev))
	b = tf.Variable( tf.truncated_normal([neuron], stddev=dev))
	return tf.matmul(inc,W) + b
class seq:

	def __init__( self , x , y ,input_shape):
		self.now_param = [input_shape]
		self.x  , self.now_output = x , x
		self.y , self.input_shape = y , input_shape
		

	def add_FC(self,size):
		self.now_output = FC_layer(self.input_shape,size,self.now_output)
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
		return self.zero_norm_train_ste

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

	def get_grad_norm(self , x , y):
		return self.sess.run(self.grad_norm2,feed_dict={
				self.x : x,
				self.y : y
			})
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
		
		# 更新直线和x轴（用一个新的x轴的标签）。
		# 用元组（Tuple）的形式返回在这一帧要被重新绘图的物体
		self.pred_line.set_xdata(self.storage[i][0])
		self.pred_line.set_ydata(self.storage[i][1])
		self.ax.set_xlabel(label)
		return self.pred_line, self.ax

	def output(self ,path='line.gif' , inter = 80):
		anim = FuncAnimation(self.fig, self.update, 
			frames=np.arange(0, len(self.storage)), interval=inter)
		anim.save(path, dpi=80, writer='imagemagick')
		plt.show()