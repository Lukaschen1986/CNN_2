# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
#os.getcwd()
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.examples.tutorials.mnist import input_data

file = "./MNIST"
mnist = input_data.read_data_sets(file, one_hot=True)

x_train = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="x_train")
y_train = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="y_train")
keep_prob = tf.placeholder(dtype=tf.float32)

def add_layer(x, in_size, out_size, active_func=None):
    global keep_prob
    # w
    w = tf.truncated_normal(shape=[in_size, out_size], mean=0.0, stddev=1.0) * 0.01
    w = tf.Variable(w, dtype=tf.float32, name="w")
    # b
    b = tf.zeros(shape=[1, out_size]) + 10**-8
    b = tf.Variable(b, dtype=tf.float32, name="b")
    # z
    z = tf.add(tf.matmul(x, w), b)
    z = tf.nn.dropout(z, keep_prob)
    # active_func
    if active_func is None:
        a = z
    else:
        a = active_func(z)
    return a


def get_accu(x, y):
    global output
    y_hat = sess.run(output, feed_dict={x_train: x})
    y_pred = tf.equal(tf.argmax(y_hat, axis=1), tf.argmax(y, axis=1))
    accu = tf.reduce_mean(tf.cast(y_pred, tf.float32))
    res = sess.run(accu, feed_dict={x_train: x, y_train: y})
    return res


layer_1 = add_layer(x=x_train, in_size=int(x_train.shape[1]), out_size=128, active_func=tf.nn.relu)
layer_2 = add_layer(x=layer_1, in_size=int(layer_1.shape[1]), out_size=64, active_func=tf.nn.relu)
output = add_layer(x=layer_2, in_size=int(layer_2.shape[1]), out_size=10, active_func=tf.nn.softmax)

loss = -tf.reduce_mean(tf.reduce_sum(y_train * tf.log(output), reduction_indices=1), reduction_indices=0)
#loss = -tf.reduce_mean(y_train * tf.log(output), reduction_indices=[1,0])
opti = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=10**-8)
#opti = tf.train.GradientDescentOptimizer(learning_rate=0.5)
objt = opti.minimize(loss)

init = tf.global_variables_initializer() # 初始化
sess = tf.Session() # 创建会话
#merged = tf.summary.merge_all() # 合并图层
#writer = tf.summary.FileWriter(logdir="D:/my_project/Python_Project/deep_learning/tf_learn/", graph=sess.graph)
sess.run(init) # 激活会话

for epoch in range(2000):
#    sess.run(fetches=objt, feed_dict={x_train: x, y_train: y})
    batch = mnist.train.next_batch(256)
    sess.run(objt, feed_dict={x_train: batch[0], y_train: batch[1], keep_prob: 0.8})
    if epoch % 50 == 0:
#        res = sess.run(fetches=merged, feed_dict={x_train: x, y_train: y})
#        writer.add_summary(summary=res, global_step=epoch)
        print(epoch, sess.run(loss, feed_dict={x_train: batch[0], y_train: batch[1], keep_prob: 0.8}))

res = get_accu(x=mnist.train.images, y=mnist.train.labels)

sess.close() 
