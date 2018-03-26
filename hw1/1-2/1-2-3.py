import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import seq , gif_gen
from matplotlib.animation import FuncAnimation
from numpy import linalg as la
tf.set_random_seed(777)



########### Function Preparing ############
def function(x):
	return np.sin(x*x/10)+x/10

def_range = (0,5)
print('{0:-^40s}'.format("first generate"))
allX = np.arange(def_range[0],def_range[1],0.001)
allY = function(allX)
	
def next_batch(size,range=(0,1)):
	b = np.random.randint(0,len(allX),size)
	return allX[b].reshape(-1,1) , allY[b].reshape(-1,1)


############ Model Preparing ################
x = tf.placeholder( tf.float32 , [None , 1])
y_ = tf.placeholder( tf.float32 , [None, 1])

model = seq(x , y_ , 1)

for _ in range(1):
	model.add_FC(4)
	model.add_activate(tf.nn.relu)


model.add_FC(1)
sess = tf.Session()
pred_y_ , train_step = model.get_train(sess, needs_Hessians= True)
zero_norm_train_step = model.get_zero_norm_train()
print("\n{0:-^40s}\n".format("all param:" + str(model.summary())))
tf.global_variables_initializer().run(session=sess)


writer = tf.summary.FileWriter("/tmp/tensorflow/1-3", sess.graph)



############## Start Training ##################
error_table = []
#model.load_model('model_1-3')
NANI = False
if not NANI:
	for _ in range(100):
		trainX , trainY = next_batch(10000,range=def_range)
		sess.run(train_step,feed_dict={
				x : trainX,
				y_ : trainY
			})
		if _ % 100 == 0 :
			loss = model.get_loss(allX.reshape(-1,1), allY.reshape(-1,1))
			error_table.append([_ , loss])
			print ("epoch %d loss %8g " %(_,loss))
if not NANI:
	for _ in range(10):
		trainX , trainY = next_batch(10000,range=def_range)
		sess.run(zero_norm_train_step,feed_dict={
				x : trainX,
				y_ : trainY
			})

		if _ % 100 == 0 :
			grad_norm = model.get_grad_norm(trainX,trainY)
			grad_hessians = model.get_grad_hessians(trainX,trainY)
			for mat in grad_hessians:
				print(mat.shape)
				#print(la.eig(mat))
			#print(grad_hessians )
			print ("epoch %d grad_norm %8g " %(_,grad_norm))

#model.save_model('model_1-3')
############### Output Prepare ###################
pred_y = sess.run(pred_y_ , feed_dict={x:allX.reshape(-1,1)}).reshape(-1)
allX = allX.reshape(-1)
allY = allY.reshape(-1)
plt.xlim(def_range)
plt.scatter(x=allX, y = allY , s = 1 , color='red')
plt.scatter(x=allX, y = pred_y , s = 1 , color='blue')
#plt.savefig('final_pic.png')
#plt.show()
	