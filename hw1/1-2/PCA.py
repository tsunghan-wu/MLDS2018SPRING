import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
param1 = np.load('output_par.npy')
param2 = np.load('output_par7.npy')
param = np.append(param1,param2 , axis=0)
# PCA
'''
pca = PCA(2)
pca.fit(param)
out = pca.transform(param)
print(out.shape)

#for row in out:
#	plt.scatter(row[0],row[1],color='red',s=1)
'''
# auto encoder

def tf_var(shape,dev=0.1):
	return tf.Variable( tf.truncated_normal(shape, stddev=dev))

# param = 8070 
data = tf.placeholder(tf.float32, [None, 8070])

## encode ##
w1 = tf_var([8070,1000])
b1 = tf_var([1000])
h1 = tf.matmul(data,w1) + b1

w2 = tf_var([1000,80])
b2 = tf_var([80])
h2 = tf.matmul(h1,w2) + b2

w3 = tf_var([80,2])
b3 = tf_var([2])

code = tf.matmul(h2,w3) + b3

## decode ## 
decode = tf.matmul(tf.matmul ( ( tf.matmul( (code-b3) , tf.transpose(w3)) - b2 ),tf.transpose(w2) - b1 ) , tf.transpose(w1))


loss = tf.reduce_mean(tf.square( decode - data))		
train_step = tf.train.AdamOptimizer(1e-5).minimize(loss)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

for _ in range(1000):
	train_step.run(session = sess , feed_dict={
			data : param
		})
	if _ % 10 == 0:
		print('itr:' , _ , ',' , loss.eval(session=sess,feed_dict={data:param}))

p1 = code.eval(session=sess,feed_dict={data:param1})
p2 = code.eval(session=sess,feed_dict={data:param2})

for row in p1:
	plt.scatter(row[0],row[1],color='blue',s=1)

for row in p2:
	plt.scatter(row[0],row[1],color='red',s=1)

plt.show()