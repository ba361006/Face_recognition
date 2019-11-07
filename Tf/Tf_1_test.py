import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*2.5 + 7

Weights = tf.Variable(tf.random_normal([1],-3.,3.))
biases = tf.Variable(tf.zeros([1]) + 0.1) #biases要盡量>0

y = x_data*Weights + biases


loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step,sess.run(Weights),sess.run(biases))

