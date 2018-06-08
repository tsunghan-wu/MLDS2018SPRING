import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
from skimage import io
from util import *


def discriminator(inputs, mode, reuse):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        kernel_size = 5
        dc1 = batch_norm(conv2d('d_conv1', tensor=inputs, ksize=kernel_size, out_dim=32), is_training=mode, name="d_bn1")
        dc1 = tf.nn.leaky_relu(dc1, alpha=0.1)
        dc2 = batch_norm(conv2d('d_conv2', tensor=dc1, ksize=kernel_size, out_dim=64), is_training=mode, name="d_bn2")
        dc2 = tf.nn.leaky_relu(dc2, alpha=0.1)
        dc3 = batch_norm(conv2d('d_conv3', tensor=dc2, ksize=kernel_size, out_dim=128), is_training=mode, name="d_bn3")
        dc3 = tf.nn.leaky_relu(dc3, alpha=0.1)
        dc4 = batch_norm(conv2d('d_conv4', tensor=dc3, ksize=kernel_size, out_dim=256), is_training=mode, name="d_bn4")
        dc4 = tf.nn.leaky_relu(dc4, alpha=0.1)
        df1 = tf.reshape(dc4, [-1, 4*4*256])
        label = fully_connected('d_last_dense', df1, 1)
        return label    


def generator(batch_size, mode, reuse=False):
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        noise = tf.random_normal([batch_size, 100])

        h1 = batch_norm(fully_connected('g_fc_1', noise, 4*4*256), is_training=mode, name="g_bn1")
        h1 = tf.nn.leaky_relu(h1, alpha=0.1)
        c0 = tf.reshape(h1, [batch_size, 4, 4, 256])
        c1 = batch_norm(deconv2d('g_deconv1', tensor=c0, ksize=4, outshape=[batch_size, 8, 8, 256]), is_training=mode, name="g_bn2")
        c1 = tf.nn.leaky_relu(c1, alpha=0.1)
        c2 = batch_norm(deconv2d('g_deconv2', tensor=c1, ksize=4, outshape=[batch_size, 16, 16, 128]), is_training=mode, name="g_bn3")
        c2 = tf.nn.leaky_relu(c2, alpha=0.1)
        c3 = batch_norm(deconv2d('g_deconv3', tensor=c2, ksize=4, outshape=[batch_size, 32, 32, 64]), is_training=mode, name="g_bn4")
        c3 = tf.nn.leaky_relu(c3, alpha=0.1)
        c4 = deconv2d('g_deconv4', tensor=c3, ksize=4, outshape=[batch_size, 64, 64, 3])
        fake_img = tf.nn.tanh(c4)
        return fake_img

def train(X, batch_size=32, flog=sys.stdout):
    with tf.variable_scope(tf.get_variable_scope()):
        z = tf.placeholder(dtype=tf.float32, shape=[batch_size, 100])
        real_data = tf.placeholder(tf.float32, shape=[batch_size,64, 64, 3])
        isTrain = tf.placeholder(tf.bool, shape=())
        with tf.variable_scope(tf.get_variable_scope()):
            fake_data = generator(batch_size, isTrain)
            real_label = discriminator(real_data, isTrain, reuse=False)
            fake_label = discriminator(fake_data, isTrain, reuse=True)

        d_vars = tf.trainable_variables('discriminator')
        g_vars = tf.trainable_variables('generator')
        # loss functrion
        gen_loss = -tf.reduce_mean(fake_label)
        dis_loss = tf.reduce_mean(fake_label) - tf.reduce_mean(real_label)

        # WGAN-GP
        ## 1. interpolation
        alpha = tf.random_uniform(shape=[batch_size, 1],minval=0.,maxval=1.)
        difference = fake_data - real_data
        difference = tf.reshape(difference, [-1, 64*64*3])
        interpolation = tf.reshape(real_data, [-1, 64*64*3]) + tf.multiply(alpha, difference)
        interpolation = tf.reshape(interpolation, [-1, 64, 64, 3])
        ## 2. gradient penalty
        gradients = tf.gradients(discriminator(interpolation, isTrain, reuse=True), [interpolation])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        ## 3. append it to loss function
        dis_loss += (10 * gradient_penalty)

        
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            gen_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(gen_loss,var_list=g_vars)
            dis_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(dis_loss,var_list=d_vars)


        EPOCH = 240
        iterations = 1000
        saver = tf.train.Saver()
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            # saver.restore(sess, "./model4/epoch119/wgan_gp.ckpt")

            for epoch in range(EPOCH):
                for iters in range(iterations):
                    # get next batch
                    for dis_iter in range(4):
                        idx = np.random.randint(0, X.shape[0], batch_size)
                        img = X[idx]
                        _, d_loss = sess.run([dis_train_op, dis_loss], feed_dict={real_data: img, isTrain:True})
                    _, g_loss = sess.run([gen_train_op, gen_loss], feed_dict={isTrain:True})
                    print ("iterations = ", iters, "discriminator loss = ", d_loss, "generator loss = ", g_loss, file=flog)
                    flog.flush()
                # sample per 100 iterations
                with tf.variable_scope(tf.get_variable_scope()):
                    samples = generator(25, isTrain, reuse=True)
                    gen_imgs = sess.run(samples, feed_dict={isTrain: False})
                    # np.save("epoch"+str(epoch)+"output.npy", gen_imgs)
                    save_imgs(gen_imgs, epoch)
                # save model per 100 iterations
                if epoch > 20:
                    checkpoint_path = "./ultimate_model/epoch"+str(epoch)+"/wgan_gp.ckpt"
                    saver.save(sess, checkpoint_path)
                    
def save_imgs(gen_imgs, epoch):
    gen_imgs = (gen_imgs * 127.5 + 127.5).astype(np.uint8)
    r, c = 5, 5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("./ultimate_result/epoch"+str(epoch)+"output.png")
    plt.close()


def read_data(img_path):
    fname = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    whole_img = np.array([io.imread(x) for x in fname]).astype(np.float64)
    whole_img = (whole_img - 127.5) / 127.5
    return whole_img

if __name__ == '__main__':
    # generator()
    # discriminator()
    tf.set_random_seed(127)
    np.random.seed(127)
    input_dir = "./extra_data/images"
    img = read_data(input_dir)
    #input_dir2 = "./faces"
    #img2 = read_data(input_dir2)
    #img = np.concatenate([img1, img2], axis=0)
    log_file = open("./logs.txt", "w")
    train(img, batch_size=32, flog=log_file)





