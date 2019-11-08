import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) -0.5 + noise

with tf.name_scope('inputs'):#把這個縮排內的變數包在名為'inputs'的泡泡裡面
    xs = tf.placeholder(tf.float32,[None, 1], name = 'x_input')#命名xs這個輸入為x_input
    ys = tf.placeholder(tf.float32,[None, 1], name = 'y_input')

def add_layer(inputs, in_size, out_size, activation_function = None):

    with tf.name_scope('layer'):
            with tf.name_scope('Weights'):
                Weights = tf.Variable(tf.random_normal([in_size, out_size]))
            with tf.name_scope('baises'):
                biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            with tf.name_scope('Wx_plus_b'):
                Wx_plus_b = tf.matmul(inputs, Weights) + biases
        
            if activation_function is None:
              outputs = Wx_plus_b
            else:
                outputs = activation_function(Wx_plus_b)
            return outputs


l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function = None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))

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
    for i in range(1001):
        sess.run(train_step, feed_dict={ys: y_data, xs: x_data})
        if i % 50 == 0:
            
            print("i = ", i, sess.run(loss, feed_dict={ys: y_data, xs: x_data},))

            try:
                ax.lines.remove(lines[0])

            except Exception:
                pass
            
            prediction_value = sess.run(prediction, feed_dict={xs: x_data})

            lines = ax.plot(x_data, prediction_value,'r-', lw = 5)

            plt.pause(0.1) 

writer = tf.summary.FileWriter("log_1",tf.get_default_graph()) #把參數下載到log_1資料夾李
writer.close()

#開啟tensorboard方法: 到有tensorboard的資料夾空白處按下shift + 滑鼠右鍵開啟powershell
#開啟後在CMD輸入.\tensorboard --D:\Tools\Tensorflow_projects\Test1_MNIST_VSCode\log_1 (參數所在的資料夾)
#在瀏覽器貼上CMD裡的網址即可