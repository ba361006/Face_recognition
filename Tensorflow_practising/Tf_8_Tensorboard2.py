import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) -0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None, 1], name = 'x_input')
    ys = tf.placeholder(tf.float32,[None, 1], name = 'y_input')

def add_layer(inputs, in_size, out_size,n_layer, activation_function = None):
    layer_name = 'layer%s'% n_layer # %s是表示一個字串，也就是帶入n_layer
    with tf.name_scope('layer'):
            with tf.name_scope('Weights'):
                Weights = tf.Variable(tf.random_normal([in_size, out_size]))# tf.summary.XXX(自定義命名, 要觀察的變數)主要是用來觀察變數在訓練過程中的變化的工具有histogram(分析整群資料的分布狀況)、scalar(分析一個純量的變化)、image(圖片)
                tf.summary.histogram(layer_name+'/Weights', Weights)#用tf.summary.histogram 紀錄Weights的變化，並命名他為layer_name/Weights
            with tf.name_scope('baises'):
                biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
                tf.summary.histogram(layer_name+'/biases', biases)
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.matmul(inputs, Weights) + biases
        
            if activation_function is None:
              outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
                tf.summary.histogram(layer_name+'/outputs', outputs)
            return outputs


l1 = add_layer(xs, 1, 10, n_layer=1, activation_function = tf.nn.relu)
prediction = add_layer(l1, 10, 1,n_layer = 2, activation_function = None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init = tf.initialize_all_variables()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()


with tf.Session() as sess:
    sess.run(init)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("log_1",sess.graph)
    for i in range(1001):
        sess.run(train_step, feed_dict={ys: y_data, xs: x_data})
        if i % 50 == 0:
            result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
            writer.add_summary(result,i)
            print("i = ", i, sess.run(loss, feed_dict={ys: y_data, xs: x_data}))

            try:
                ax.lines.remove(lines[0])

            except Exception:
                pass
            
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})

            lines = ax.plot(x_data, prediction_value,'r-', lw = 5)

            plt.pause(0.1) 

