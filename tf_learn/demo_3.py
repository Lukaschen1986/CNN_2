# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
#import numpy as np
#import pandas as pd
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
import matplotlib.pyplot as plt
import os
os.getcwd()
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
#from sklearn.preprocessing import OneHotEncoder
#from keras.models import Model
#from keras.layers import Input, Lambda

# load data
digits = load_digits()
print(digits.images[0])
x = digits.data
x = x.reshape(1797, 8, 8, 1) # NHWC
y = digits.target

mapper = LabelBinarizer()
mapper_fit = mapper.fit(y)
y = mapper_fit.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_train.shape[1]
y_train.shape[1]

# placeholder
features = tf.placeholder(tf.float32, shape=[None, 64])
features = tf.reshape(features, shape=[-1,8,8,1], name="features")

labels = tf.placeholder(tf.float32, shape=[None, 10], name="labels")
keep_prob = tf.placeholder(tf.float32, name="keep_prob")

# conv layer
def conv_layer(x, w_shape, b_shape, cs, active_func, ks, ps, 
               w_name, b_name, z_name, a_name, p_name, 
               w_lam, lam_name,
               use_bn=True):
    # w
    w = tf.truncated_normal(shape=w_shape, mean=0.0, stddev=0.01)
    w = tf.Variable(w, dtype=tf.float32, name=w_name)
    # L2
    tf.add_to_collection(name=lam_name, value=l2_regularizer(w_lam)(w))
    
    if use_bn:
        # z
        z = tf.nn.conv2d(input=x, filter=w, strides=[1,cs,cs,1], padding="SAME")
        # bn
        mu, var = tf.nn.moments(z, axes=[0])
        H, W, C = mu.shape
        beta = tf.zeros(shape=[H, W, C])
        beta = tf.Variable(beta, dtype=tf.float32)
        gamma = tf.ones(shape=[H, W, C])
        gamma = tf.Variable(gamma, dtype=tf.float32)
        z = tf.nn.batch_normalization(z, mean=mu, variance=var, offset=beta, scale=gamma, variance_epsilon=10**-8, name=z_name)
    else:
        # b
        b = tf.add(tf.zeros(shape=b_shape), 10**-8)
        b = tf.Variable(b, dtype=tf.float32, name=b_name)
        # z
        z = tf.nn.conv2d(input=x, filter=w, strides=[1,cs,cs,1], padding="SAME")
        z = tf.nn.bias_add(z, b, name=z_name)
    # a
    a = active_func(z, name=a_name)
    # p
    p = tf.nn.max_pool(value=a, ksize=[1,ks,ks,1], strides=[1,ps,ps,1], padding="SAME", name=p_name)
    return p
layer_1 = conv_layer(x=features, w_shape=[3,3,1,32], b_shape=[32], cs=1, active_func=tf.nn.relu, ks=2, ps=2,
                     w_name="w_1", b_name="b_1", z_name="z_layer_1", a_name="a_layer_1", p_name="p_layer_1", 
                     w_lam=0.001, lam_name="lam_1", use_bn=True)
layer_2 = conv_layer(x=layer_1, w_shape=[3,3,32,64], b_shape=[64], cs=1, active_func=tf.nn.relu, ks=2, ps=2,
                     w_name="w_2", b_name="b_2", z_name="z_layer_2", a_name="a_layer_2", p_name="p_layer_2", 
                     w_lam=0.001, lam_name="lam_2", use_bn=True)
#conv_layer_1 = Lambda(conv_layer, arguments={"w_shape":[3,3,1,32], "b_shape":[32], "cs":1, 


def flatten(x, f_name):
    _, H, W, C = x.shape
    x_flatten = tf.reshape(x, shape=[-1,int(H*W*C)], name=f_name)
    return x_flatten
flatten_layer = flatten(x=layer_2, f_name="f_layer")
#flatten_layer = Lambda(flatten)(conv_layer_2)


# dense layer
def dense_layer(x, in_size, out_size, active_func, 
                w_name, b_name, z_name, a_name, 
                w_lam, lam_name, use_bn=True):
    global keep_prob
    # w
    w = tf.truncated_normal(shape=[in_size, out_size], mean=0.0, stddev=0.01)
    w = tf.Variable(w, dtype=tf.float32, name=w_name)
    # L2
    tf.add_to_collection(name=lam_name, value=l2_regularizer(w_lam)(w))
    
    if use_bn:
        # z
        z = tf.matmul(x, w)
        # bn
        mu, var = tf.nn.moments(z, axes=[0])
        N = mu.shape[0]
        beta = tf.zeros(shape=[N])
        beta = tf.Variable(beta, dtype=tf.float32)
        gamma = tf.ones(shape=[N])
        gamma = tf.Variable(gamma, dtype=tf.float32)
        z = tf.nn.batch_normalization(z, mean=mu, variance=var, offset=beta, scale=gamma, variance_epsilon=10**-8, name=z_name)        
    else:
        # b
        b = tf.add(tf.zeros(shape=[1, out_size]), 10**-8)
        b = tf.Variable(b, dtype=tf.float32, name=b_name)
        # z
        z = tf.nn.bias_add(tf.matmul(x, w), b, name=z_name)
    # active_func
    if active_func is None:
        a = tf.nn.dropout(z, keep_prob, name=a_name)
    else:
        d = tf.nn.dropout(z, keep_prob)
        a = active_func(d, name=a_name)
    return a
layer_3 = dense_layer(x=flatten_layer, in_size=int(flatten_layer.shape[1]), out_size=512, active_func=tf.nn.relu,
                      w_name="w_3", b_name="b_3", z_name="z_layer_3", a_name="a_layer_3", 
                      w_lam=0.001, lam_name="lam_3", use_bn=True)
layer_4 = dense_layer(x=layer_3, in_size=int(layer_3.shape[1]), out_size=1024, active_func=tf.nn.relu,
                      w_name="w_4", b_name="b_4", z_name="z_layer_4", a_name="a_layer_4", 
                      w_lam=0.001, lam_name="lam_4", use_bn=True)
outputs = dense_layer(x=layer_4, in_size=int(layer_4.shape[1]), out_size=10, active_func=tf.nn.softmax,
                      w_name="w_5", b_name="b_5", z_name="z_layer_5", a_name="a_layer_5", 
                      w_lam=0.001, lam_name="lam_5", use_bn=True)
#dense_layer_1 = Lambda(dense_layer, arguments={"in_size": int(flatten_layer.shape[1]), "out_size": 512, "active_func": tf.nn.relu, "use_bn":True})(flatten_layer)

loss = -tf.reduce_mean(tf.reduce_sum(labels * tf.log(outputs), reduction_indices=1), reduction_indices=0)
#loss = -tf.reduce_mean(y_train * tf.log(output), reduction_indices=[1,0])
opti = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=10**-8)
#opti = tf.train.GradientDescentOptimizer(learning_rate=0.05)
objt = opti.minimize(loss)

init = tf.global_variables_initializer() # 初始化
saver = tf.train.Saver() # 创建保存
sess = tf.Session() # 创建会话
#merged = tf.summary.merge_all() # 合并图层
#train_writer = tf.summary.FileWriter(logdir="D:/my_project/Python_Project/deep_learning/tf_learn/train", graph=sess.graph)
#test_writer = tf.summary.FileWriter(logdir="D:/my_project/Python_Project/deep_learning/tf_learn/test", graph=sess.graph)
sess.run(init) # 激活会话

kp = 0.5; loss_train_res = []; loss_test_res = []
for epoch in range(2000):
    sess.run(objt, feed_dict={features: x_train, labels: y_train, keep_prob: kp})
    if epoch % 50 == 0:
        loss_train = sess.run(loss, feed_dict={features: x_train, labels: y_train, keep_prob: kp})
        loss_test = sess.run(loss, feed_dict={features: x_test, labels: y_test, keep_prob: 1.0})
        print("epoch: %d; loss_train: %.5g; loss_test: %.5g" % (epoch, loss_train, loss_test))
        loss_train_res.append(loss_train)
        loss_test_res.append(loss_test)

save_path = saver.save(sess, save_path="./sess.ckpt") # ".ckpt" 扩展名表示"checkpoint"

# check plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(loss_train_res, color="blue", linewidth=1, linestyle="dashed")
ax.plot(loss_test_res, color="red", linewidth=2)
plt.show()

# y_pred
y_hat = sess.run(outputs, feed_dict={features: x_test, keep_prob: 1.0})
y_pred = tf.equal(tf.argmax(y_hat, axis=1), tf.argmax(y_test, axis=1))
accu = tf.reduce_mean(tf.cast(y_pred, tf.float32))
res = sess.run(accu, feed_dict={features: x_test, labels: y_test, keep_prob: 1.0})   
print(res)
sess.close()

# load sess 
'''
因为 tf.train.Saver.restore() 设定了 TensorFlow 变量，这里你不需要调用 tf.global_variables_initializer()了
'''
saver = tf.train.Saver() # 创建保存
sess = tf.Session() # 创建会话
saver.restore(sess, save_path="./sess.ckpt")
print()

# 抽取某一层特征
y_flatten_layer = sess.run(flatten_layer, feed_dict={features: x_test, keep_prob: 1.0})

# 抽取某一层权重
reader = tf.train.NewCheckpointReader(filepattern="./sess.ckpt")
all_variables = reader.get_variable_to_shape_map()
w_1 = reader.get_tensor("w_1")

# 移除先前的权重和偏置项
tf.reset_default_graph()
