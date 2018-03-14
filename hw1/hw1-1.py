import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from util import seq , gif_gen
from matplotlib.animation import FuncAnimation



########### Function Preparing ############
f_a = 0.5
f_b = 2
def function(x):
	'''
	#function 1
	y = x * 0
	for n in range(10):
		y += np.power(f_a ,n) * np.cos( np.power(f_b,n) * np.pi * x)
	return y
	'''
	#funciton2
	return np.sin(x) + np.cos(x*x)

def_range = (0,10)
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

choose_model = 2
if choose_model == 1:
	model.add_FC(3450)
	model.add_activate(tf.nn.relu)
elif choose_model == 2:
	for layer in [67,70,76]:
		model.add_FC(layer)
		model.add_activate(tf.nn.relu)
else:
	for _ in range(5):
		model.add_FC(50)
		model.add_activate(tf.nn.relu)


model.add_FC(1)
sess = tf.Session()
pred_y_ , train_step = model.get_train(sess)
print("\n{0:-^40s}\n".format("all param:" + str(model.summary())))
tf.global_variables_initializer().run(session=sess)

############## Start Training ##################

#model.load_model('model_5layers')
#model.load_model('model2')
#model.load_model('model_1layers_test')
GIF = gif_gen(def_range[0],def_range[1],0.00001 , allX , function(allX))
error_table = []
for _ in range(200000):
	trainX , trainY = next_batch(10000,range=def_range)
	sess.run(train_step,feed_dict={
			x : trainX,
			y_ : trainY
		})
	if _ % 100 == 0 :
		loss = model.get_loss(allX.reshape(-1,1), allY.reshape(-1,1))
		error_table.append([_ , loss])
		print ("epoch %d loss %8g " %(_,loss))
	if _ % 500 ==  0 : 
		GIF.set_frame(x = allX.reshape(-1) , y = sess.run(pred_y_ , feed_dict={x:allX.reshape(-1,1)}))
		
model.save_model('model_3layers')
############### Output Prepare ###################
import csv 
cout = csv.writer(open('error_table_3layers.csv' , 'w'))
cout.writerows(error_table)

pred_y = sess.run(pred_y_ , feed_dict={x:allX.reshape(-1,1)}).reshape(-1)
allX = allX.reshape(-1)
allY = allY.reshape(-1)

GIF.output()
plt.scatter(x=allX, y = allY , s = 1 , color='red')
plt.scatter(x=allX, y = pred_y , s = 1 , color='blue')
plt.savefig('final_pic.png')

	