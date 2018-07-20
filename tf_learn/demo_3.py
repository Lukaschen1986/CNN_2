# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
#from sklearn.preprocessing import OneHotEncoder

# load data
digits = load_digits()
print(digits.images[0])
x = digits.data
x = x.reshape(1797, 8, 8, 1) # NHWC
y = digits.target

mapper = LabelBinarizer()
mapper_fit = mapper.fit(y)
y = mapper_fit.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_train.shape[1]
y_train.shape[1]

# placeholder
x_inputs = tf.placeholder(tf.float32, shape=[None, 64], name="x_train")
x_inputs = tf.reshape(x_inputs, shape=[-1,8,8,1])

y_inputs = tf.placeholder(tf.float32, shape=[None, 10], name="y_train")
keep_prob = tf.placeholder(tf.float32)


# conv layer
def conv_layer(x, w_shape, b_shape, cs, active_func, ks, ps):
    # w
    w = tf.truncated_normal(shape=w_shape, mean=0.0, stddev=0.01)
    w = tf.Variable(w, dtype=tf.float32, name="w")
    # b
    b = tf.add(tf.zeros(shape=b_shape), 10**-8)
    b = tf.Variable(b, dtype=tf.float32, name="b")
    # z
    z = tf.nn.conv2d(input=x, filter=w, strides=[1,cs,cs,1], padding="SAME")
    z = tf.add(z, b)
    # a
    a = active_func(z)
    # p
    p = tf.nn.max_pool(value=a, ksize=[1,ks,ks,1], strides=[1,ps,ps,1], padding="SAME")
    return p
conv_layer_1 = conv_layer(x=x_inputs, w_shape=[3,3,1,32], b_shape=[32], cs=1, active_func=tf.nn.relu, ks=2, ps=2)
conv_layer_2 = conv_layer(x=conv_layer_1, w_shape=[3,3,32,64], b_shape=[64], cs=1, active_func=tf.nn.relu, ks=2, ps=2)
conv_layer_2.shape

def flatten(x):
    _, H, W, C = x.shape
    x_flatten = tf.reshape(x, shape=[-1,int(H*W*C)])
    return x_flatten
flatten_layer = flatten(x=conv_layer_2)

# dense layer
def dense_layer(x, in_size, out_size, active_func=None):
    global keep_prob
    # w
    w = tf.truncated_normal(shape=[in_size, out_size], mean=0.0, stddev=0.01)
    w = tf.Variable(w, dtype=tf.float32, name="weights")
    # b
    b = tf.add(tf.zeros(shape=[1, out_size]), 10**-8)
    b = tf.Variable(b, dtype=tf.float32, name="biases")
    # z
    z = tf.add(tf.matmul(x, w), b)
    z = tf.nn.dropout(z, keep_prob)
    # active_func
    if active_func is None:
        a = z
    else:
        a = active_func(z)
    return a
dense_layer_1 = dense_layer(x=flatten_layer, in_size=int(flatten_layer.shape[1]), out_size=512, active_func=tf.nn.relu)
dense_layer_2 = dense_layer(x=dense_layer_1, in_size=int(dense_layer_1.shape[1]), out_size=1024, active_func=tf.nn.relu)
outputs = dense_layer(x=dense_layer_2, in_size=int(dense_layer_2.shape[1]), out_size=10, active_func=tf.nn.softmax)

loss = -tf.reduce_mean(tf.reduce_sum(y_inputs * tf.log(outputs), reduction_indices=1), reduction_indices=0)
#loss = -tf.reduce_mean(y_train * tf.log(output), reduction_indices=[1,0])
#opti = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=10**-8)
opti = tf.train.GradientDescentOptimizer(learning_rate=0.05)
objt = opti.minimize(loss)

x_inputs = Input(shape=((x_inputs.shape[1], x_inputs.shape[2], x_inputs.shape[3])), dtype=np.float32, name="input_layer")
model = Model(inputs=x_inputs, outputs=y_inputs)

init = tf.global_variables_initializer() # 初始化
saver = tf.train.Saver() # 创建保存
sess = tf.Session() # 创建会话
#merged = tf.summary.merge_all() # 合并图层
#train_writer = tf.summary.FileWriter(logdir="D:/my_project/Python_Project/deep_learning/tf_learn/train", graph=sess.graph)
#test_writer = tf.summary.FileWriter(logdir="D:/my_project/Python_Project/deep_learning/tf_learn/test", graph=sess.graph)
sess.run(init) # 激活会话

kp = 0.8; loss_train_res = []; loss_test_res = []
for epoch in range(2000):
    sess.run(objt, feed_dict={x_inputs: x_train, y_inputs: y_train, keep_prob: kp})
    if epoch % 50 == 0:
        loss_train = sess.run(loss, feed_dict={x_inputs: x_train, y_inputs: y_train, keep_prob: kp})
        loss_test = sess.run(loss, feed_dict={x_inputs: x_test, y_inputs: y_test, keep_prob: 1.0})
        print("epoch: %d; loss_train: %.5g; loss_test: %.5g" % (epoch, loss_train, loss_test))
        loss_train_res.append(loss_train)
        loss_test_res.append(loss_test)

save_path = saver.save(sess, save_path="sess.ckpt")

# check plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(loss_train_res, color="red", linewidth=1)
ax.plot(loss_test_res, color="blue", linewidth=1, linestyle="dashed")
plt.show()

# y_pred
y_hat = sess.run(outputs, feed_dict={x_inputs: x_test, keep_prob: 1.0})
y_pred = tf.equal(tf.argmax(y_hat, axis=1), tf.argmax(y_test, axis=1))
accu = tf.reduce_mean(tf.cast(y_pred, tf.float32))
res = sess.run(accu, feed_dict={x_inputs: x_test, y_inputs: y_test, keep_prob: 1.0})   
print(res)
sess.close() 

# load sess
init = tf.global_variables_initializer() # 初始化
saver = tf.train.Saver() # 创建保存
sess = tf.Session() # 创建会话
sess.run(init)
saver.restore(sess, save_path="sess.ckpt")
print()
