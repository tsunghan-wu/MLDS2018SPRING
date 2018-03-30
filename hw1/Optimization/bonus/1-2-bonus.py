import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



########### Function Preparing ############
def function(x):
	return np.sin(x) + np.cos(x*x/5)

def_range = (-5,5)
print('{0:-^40s}'.format("first generate"))
allX = np.arange(def_range[0],def_range[1],0.00001)
allY = function(allX)
	
def next_batch(size,range=(0,1)):
	b = np.random.randint(0,len(allX),size)
	return allX[b].reshape(-1,1) , allY[b].reshape(-1,1)


############ Model Preparing ################
x = tf.placeholder( tf.float32 , [None , 1])
y_ = tf.placeholder( tf.float32 , [None, 1])

def cret(shape):
	return tf.Variable( tf.truncated_normal(shape, stddev=0.1))

def crep(shape):
	return tf.placeholder( tf.float32 , shape)

neuron = [1,4,4,1]
w_shape = [ [neuron[i] , neuron[i+1]] for i in range(len(neuron)-1)]
b_shape = [ [neuron[i+1]]  for i in range(len(neuron)-1)]
# p: perturb
w , b , pw , pb = [],[],[],[]
for ws , bs in zip(w_shape , b_shape):
	temp_w = cret(ws)
	temp_b = cret(bs)
	now_pw =crep(ws)
	now_pb =crep(bs)
	w.append(temp_w + now_pw)
	b.append(temp_b + now_pb)
	pw.append(now_pw)
	pb.append(now_pb)

rr = lambda size : np.random.uniform(-0.1,0.1,size) if np.random.randint(0,10) == 0 else np.zeros(size)

zero_w_dict = { now_pw: np.zeros(ws) for now_pw , ws in zip(pw,w_shape)}
zero_b_dict = { now_pb: np.zeros(bs) for now_pb , bs in zip(pb,b_shape)}
zero_dict = {}
zero_dict.update(zero_w_dict)
zero_dict.update(zero_b_dict)
out = x
for noww,nowb in zip(w,b):
	out = tf.nn.selu(tf.matmul(out,noww) + nowb)
y =  out
 
sess = tf.Session()
loss = tf.nn.l2_loss(y-y_)*2
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

tf.global_variables_initializer().run(session=sess)
############## Start Training ##################
parameters_table = []
loss_table = []
color_table = []
for _ in range(40000):
	trainX , trainY = next_batch(20000,range=def_range)
	now_feed_dict = {	x : trainX,	y_ : trainY	}
	now_feed_dict.update(zero_dict)

	sess.run(train_step,feed_dict=now_feed_dict)
	if _ % 100 == 0 and _ > 5000:
		# random sampling
		for __ in range(50):
			pw_dict = { now_pw: rr(ws) for now_pw , ws in zip(pw,w_shape)}
			pb_dict = { now_pb: rr(bs) for now_pb , bs in zip(pb,b_shape)}
			p_dict = {}
			p_dict.update(pw_dict)
			p_dict.update(pb_dict)
			p_dict.update({x:trainX, y_:trainY})
			ob_w, ob_b , eva_loss = sess.run([w,b,loss],feed_dict=p_dict)
			ob_w_flat = np.concatenate(np.array([ now_ob_w.flatten() for now_ob_w in ob_w ]),axis=0)
			ob_b_flat = np.concatenate(np.array([ now_ob_b.flatten() for now_ob_b in ob_b ]),axis=0)
			ob_flat = np.concatenate( [ob_w_flat,ob_b_flat],axis=0)
			parameters_table.append(ob_flat)
			loss_table.append(eva_loss)
			color_table.append(0)

		ob_w, ob_b , eva_loss = sess.run([w,b,loss],feed_dict=now_feed_dict)
		ob_w_flat = np.concatenate(np.array([ now_ob_w.flatten() for now_ob_w in ob_w ]),axis=0)
		ob_b_flat = np.concatenate(np.array([ now_ob_b.flatten() for now_ob_b in ob_b ]),axis=0)
		ob_flat = np.concatenate( [ob_w_flat,ob_b_flat],axis=0)
		parameters_table.append(ob_flat)
		loss_table.append(eva_loss)
		color_table.append(1)
		print ("epoch %d loss %8g " %(_,eva_loss))

np.save('bonus_params' , parameters_table)
np.save('bonus_loss' , loss_table)
np.save('bonus_colors' ,color_table)

'''
############### Output Prepare ###################
now_feed_dict = {x : allX.reshape(-1,1)}
now_feed_dict.update(zero_dict)	
pred_y = sess.run(y , feed_dict=now_feed_dict).reshape(-1)
print(pred_y.shape)
print(allX.shape)
print(allY.shape)

allX = allX.reshape(-1)
allY = allY.reshape(-1)
plt.scatter(x=allX, y = allY , s = 1 , color='red')
plt.scatter(x=allX, y = pred_y , s = 1 , color='blue')
plt.show()


saver = tf.train.Saver()
saver.save(sess, "model1-2-bonus/model.ckpt")
'''