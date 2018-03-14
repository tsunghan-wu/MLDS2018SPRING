import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def init_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def conv2d(x, W):
	return tf.nn.conv2d(
		x, W, strides=[1, 1, 1, 1], padding = 'SAME')
	# 1 [x_strides , y_strides] 1
	# padding:  output size + padding = input_size 
def max_pool_2x2(x):
	return tf.nn.max_pool(
		x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
	# 1 [x_strides , y_strides] 1
#### global parameters #### 
learning_rate = 1e-4

###########################
x = tf.placeholder( tf.float32 , [ None , 28*28 ])
y_= tf.placeholder( tf.float32 , [ None , 10 ])
### conv layer 1 ####
w_conv1 = init_weights([5,5,1,32])
b_conv1 = init_weights([32])
# 5*5 patch , input channel 1 , output channel 32

x_image = tf.reshape(x, [-1, 28, 28, 1])

# -1 means this dimension will calculate automatically.

#### pool layer 1 ####
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

### conv layer 2 ####
w_conv2 = init_weights([5,5,32,64])
b_conv2 = init_weights([64])
# 5*5 patch , input channel 32 , output channel 64

#### pool layer 2 ####
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


#### fully connected layer ####
w_fc1 = init_weights([ 7 * 7 * 64 , 1024])
# this fc layer has 1024 neurons
b_fc1 = init_weights([1024])

#### Flatten layer ####
h_flat = tf.reshape(h_pool2, [ -1 , 7 * 7 * 64 ])
h_fc1 = tf.nn.relu(tf.matmul(h_flat , w_fc1) + b_fc1)

#### dropout layer ####
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = init_weights([1024, 10])
b_fc2 = init_weights([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#### model compile ####
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
	logits =y_conv,labels =y_)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_pred = tf.equal( tf.argmax(y_conv,1) , tf.argmax(y_,1) )
acc = tf.reduce_mean( tf.cast(correct_pred , tf.float32))


sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

writer = tf.summary.FileWriter("/tmp/tensorflow/MNIST", sess.graph)

for i in range(20000):
	batch = mnist.train.next_batch(100)
	if i%100 == 0:
		train_acc = acc.eval(session = sess,feed_dict={
			x : batch[0],
			y_: batch[1],
			keep_prob: 1.0
		})
		print("step %d , acc %g" % (i,train_acc))
	train_step.run(session = sess , feed_dict={
			x : batch[0],
			y_: batch[1],
			keep_prob: 0.5
		})
print("test acc %g" % acc.eval(session = sess , feed_dict={
		x : mnist.test.images,
		y_: mnist.test.labels,
		keep_prob: 1.0
	}))
