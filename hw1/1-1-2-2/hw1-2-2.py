import tensorflow as tf
import numpy as np 
import models
#### data process ########
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

all_data = None
all_label = None
for _ in range(1,6):
	batch = unpickle("cifar-10-batches-py/data_batch_" + str(_))
	if all_data is not None:
		all_data = np.append(all_data,batch[b"data"],axis=0)
		all_label = np.append(all_label,batch[b"labels"],axis=0)
	else:
		all_data = batch[b"data"]
		all_label= batch[b"labels"]
temp = np.zeros((50000,10))
## one-hot
temp[np.arange(50000),all_label] =  1
all_label = temp

def next_batch(size=1000):
	b =  np.random.randint(0,50000,size)
	return all_data[b].reshape(size,32,32,3) , all_label[b].reshape(size,10)
	
###########################

#### global parameters #### 
learning_rate = 1e-4
###########################

x = tf.placeholder( tf.float32 , [ None , 32,32,3 ])
y_= tf.placeholder( tf.float32 , [ None , 10 ])
y = models.model3(x)


#### model compile ####
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits =y,labels =y_)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_pred = tf.equal( tf.argmax(y,1) , tf.argmax(y_,1) )
acc = tf.reduce_mean( tf.cast(correct_pred , tf.float32))

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

# for tensorboard.
# writer = tf.summary.FileWriter("/tmp/tensorflow/MNIST", sess.graph)

error_table = []
for i in range(50000):
	batch = next_batch(100)
	train_step.run(session = sess , feed_dict={
			x : batch[0],
			y_: batch[1],
		})
	train_acc,loss = sess.run( (acc,cross_entropy),feed_dict={
		x : batch[0],
		y_: batch[1],
	})
	error_table.append([i , train_acc , np.mean(loss)])
	if i%100 == 0:
		print("step:%d,\tacc:%g,\tloss:%g" % (i,train_acc,np.mean(loss)))

saver = tf.train.Saver()
saver.save(sess,"cnn3_dim_128_128_128/model.ckpt")

############### Output Prepare ###################
import csv 
cout = csv.writer(open('error_table_cnn3_dim128_128_128.csv' , 'w'))
cout.writerows(error_table)
