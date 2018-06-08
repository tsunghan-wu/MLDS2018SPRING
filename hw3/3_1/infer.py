import os
import sys
import numpy as np
import argparse
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
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

def infer(model_path, output_img_path, batch_size=25):
    with tf.variable_scope(tf.get_variable_scope()):
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

        
        # with tf.variable_scope(tf.get_variable_scope(), reuse=None):
        #     gen_train_op = tf.train.AdamOptimizer(
        #         learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(gen_loss,var_list=g_vars)
        #     dis_train_op = tf.train.AdamOptimizer(
        #         learning_rate=1e-4,beta1=0.5,beta2=0.9).minimize(dis_loss,var_list=d_vars)


        saver = tf.train.Saver()
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)
            # saver.restore(sess, "./gan_model/wgan_gp.ckpt")
            tf.set_random_seed(997)
            np.random.seed(997)
            # sample per 100 iterations
            with tf.variable_scope(tf.get_variable_scope()):
                samples = generator(25, isTrain, reuse=True)
                gen_imgs = sess.run(samples, feed_dict={isTrain: False})
                # np.save("epoch"+str(epoch)+"output.npy", gen_imgs)
                save_imgs(gen_imgs, output_img_path)
                   
def save_imgs(gen_imgs, output_img_path):
    gen_imgs = (gen_imgs * 127.5 + 127.5).astype(np.uint8)
    r, c = 5, 5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(output_img_path)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MLDS 3-1')
    parser.add_argument('--model', type=str, help='model_file')
    parser.add_argument('--out', type=str, help='output image path')
    args = parser.parse_args()    
    infer(args.model, args.out, batch_size=25)





