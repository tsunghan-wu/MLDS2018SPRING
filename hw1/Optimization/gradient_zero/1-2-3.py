import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import linalg as la

########### Function Preparing ############
def function(x):
	return np.sin(x) + np.cos(x*x/10)

def_range = (-8,8)
print('{0:-^40s}'.format("first generate"))
allX = np.arange(def_range[0],def_range[1],0.0001).reshape(-1,1)
allY = function(allX).reshape(-1,1)
	
def next_batch(size,range=(0,1)):
	b = np.random.randint(0,len(allX),size)
	return allX[b] , allY[b]

def cret(shape):
	return tf.Variable( tf.truncated_normal(shape, stddev=0.1))

############ Model Preparing ################
x = tf.placeholder( tf.float32 , [None , 1])
y_ = tf.placeholder( tf.float32 , [None, 1])
neu =  10
params = np.array([neu,neu,neu**2,neu,neu,1])
tempw1 = cret([neu])
w1 = tf.reshape(tempw1 , [1,neu])
b1 = cret( [neu] )
h1 = tf.nn.relu(tf.matmul(x,w1) + b1)

tempw2 = cret([neu**2])
w2 = tf.reshape(tempw2 , [neu,neu])
b2 = cret( [neu] )
h2 = tf.nn.relu(tf.matmul(h1,w2) + b2)

tempw3 = cret([neu])
w3 = tf.reshape(tempw3 , [neu,1])
b3 = cret( [1] )
y = tf.matmul(h2,w3) + b3



train_loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(train_loss)
grads = tf.gradients(train_loss,tf.trainable_variables())
norm = tf.norm(tf.concat([ tf.reshape(grad,[-1]) for grad in grads],axis=0),2)
norm_step = tf.train.AdamOptimizer(1e-4).minimize(norm)
hessians_pieces = tf.hessians(train_loss,tf.trainable_variables())


recorder = np.loadtxt('1-2-3recorder').tolist()
for times in range(100):
	print('{0:-^40s}'.format("now times: " +str(times)))
	sess = tf.Session()
	tf.global_variables_initializer().run(session=sess)

	############## Start Training ##################
	epoch = 15000
	for _ in range(epoch*3):
		trainX , trainY = next_batch(sc,range=def_range)
		if _ < epoch*3:
			sess.run(train_step,feed_dict={	x : trainX,	y_ : trainY	})
		else:
			sess.run(norm_step,feed_dict={	x : trainX,	y_ : trainY	})
		
		if _ % 1000 == 999 :
			loss , norm_get= sess.run( (train_loss,norm), feed_dict={
				x : allX.reshape(-1,1), 
				y_: allY.reshape(-1,1)})
			if _ < epoch*3:
				print('(minimize loss)' ,end=' ')
			else:
				print('(minimize norm)' ,end=' ')
			print ("epoch %5d , loss %.8g , norm %.8g" %(_,loss,norm_get))

	get_h = sess.run(hessians_pieces , feed_dict = {x:trainX , y_:trainY})
	H = np.zeros( (np.sum(params),np.sum(params)) )
	count = 0
	for oao in get_h:
		sh = oao.shape[0]
		for i in range(sh):
			for j in range(sh):
				H[count+i][count+j] = oao[i][j]
		count+=oao.shape[0]
	w, v = np.linalg.eig(H)
	recorder.append( (w[w>0].shape[0]/np.sum(params) , loss) )
	print('minimal ratio' , w[w>0].shape[0]/np.sum(params))
	sess.close()
	np.savetxt('1-2-3recorder' , np.array(recorder))
'''
############### Output Prepare ###################
pred_y = sess.run(y , feed_dict={x:allX})

plt.xlim(def_range)
plt.scatter(x=allX.reshape(-1), y = allY.reshape(-1) , s = 1 , color='red')
plt.scatter(x=allX.reshape(-1), y = pred_y.reshape(-1) , s = 1 , color='blue')
plt.show()
'''