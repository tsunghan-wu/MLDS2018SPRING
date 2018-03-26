import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import seq , gif_gen
from matplotlib.animation import FuncAnimation



########### Function Preparing ############
def function(x):
	return np.sin(x) + np.cos(x*x)

def_range = (-10,10)
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

for _ in range(10):
	model.add_FC(10)
	model.add_activate(tf.nn.relu)


model.add_FC(1)
sess = tf.Session()
pred_y_ , train_step = model.get_train(sess)
print("\n{0:-^40s}\n".format("all param:" + str(model.summary())))
tf.global_variables_initializer().run(session=sess)
############## Start Training ##################
error_table = []
for _ in range(200):
	trainX , trainY = next_batch(10000,range=def_range)
	sess.run(train_step,feed_dict={
			x : trainX,
			y_ : trainY
		})
	if _ % 100 == 0 :
		loss = model.get_loss(allX.reshape(-1,1), allY.reshape(-1,1))
		error_table.append([_ , loss])
		print ("epoch %d loss %8g " %(_,loss))
		
############### Output Prepare ###################
pred_y = sess.run(pred_y_ , feed_dict={x:allX.reshape(-1,1)}).reshape(-1)
allX = allX.reshape(-1)
allY = allY.reshape(-1)
plt.scatter(x=allX, y = allY , s = 1 , color='red')
plt.scatter(x=allX, y = pred_y , s = 1 , color='blue')
plt.savefig('final_pic.png')
model.save_model('model_1-3')