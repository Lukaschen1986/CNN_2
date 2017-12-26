import numpy as np
import tensorflow as tf

learning_init = 0.1
decay_rate = 0.1
global_steps = 300000
decay_steps = 70000
learning_rate = learning_init * decay_rate**(np.arange(global_steps) / decay_steps)  
plt.plot(learning_rate)


learning_init = 0.1
decay_rate = 0.9
global_step = tf.Variable(0)
global_steps = 300000
decay_steps = 70000
learning_rate = tf.train.exponential_decay(learning_init, global_step, decay_steps, decay_rate, staircase=True) # staircase=False 每一步都更新；staircase=True 每decay_steps步更新

lr_res = []
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(global_steps):
        lr = sess.run(learning_rate, feed_dict={global_step: i})
        lr_res.append(lr)
plt.plot(lr_res)
