# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import tensorflow as tf

# data
x = np.random.rand(100).astype(np.float32)
y = x * 0.1 + 0.3

# params
x_train = tf.placeholder(dtype=tf.float32, shape=[None])
y_train = tf.placeholder(dtype=tf.float32, shape=[None])
#w = tf.Variable(tf.random_uniform(shape=[1], minval=-1.0, maxval=1.0), dtype=tf.float32)
w = tf.Variable(initial_value=tf.zeros(shape=[1]), name="w", dtype=tf.float32)
b = tf.Variable(initial_value=tf.zeros(shape=[1]), name="b", dtype=tf.float32)

# predict
y_hat = tf.add(tf.multiply(x_train, w), b)

# obj func
loss = tf.reduce_mean(tf.square(y_train - y_hat))
opti = tf.train.GradientDescentOptimizer(learning_rate=0.5)
objt = opti.minimize(loss)

# solve
init = tf.global_variables_initializer() # 初始化
sess = tf.Session() # 创建会话
sess.run(init) # 激活会话

for epoch in range(200):
    sess.run(fetches=objt, feed_dict={x_train: x, y_train: y})
    if epoch % 10 == 0: # 每10步输出一次结果
        print(epoch, sess.run(w), sess.run(b))
        
sess.close()

#######################################################################################################
#init_val = tf.Variable(initial_value=tf.zeros(shape=[1]), name="init_val", dtype=tf.float32)
init_val = tf.Variable(initial_value=0.0, name="init_val", dtype=tf.float32) # 变量
add_val = tf.constant(value=1.0, name="add_val", dtype=tf.float32) # 常量
new_val = tf.add(init_val, add_val) # 增量加
update = tf.assign(init_val, new_val)

init = tf.global_variables_initializer() # 初始化
sess = tf.Session() # 创建会话
sess.run(init) # 激活会话

for epoch in range(10):
    sess.run(fetches=update)
    print(sess.run(init_val))

sess.close()
    
#######################################################################################################

