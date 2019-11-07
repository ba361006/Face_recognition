import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot=True) #讀取資料

def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuarcy(v_xs, v_ys):
    global prediction #變成全域變數  https://kaiching.org/pydoing/py/python-global.html
    y_pre = sess.run(prediction, feed_dict={xs: v_xs}) 
    #預測值，也是一行十列，但是是機率所以不一定是漂亮的數，例如在三的機率最高，那就判斷他是三

    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    #與真實數字對比，拿預測出來的值跟實際上的值對比，
    #tf.argmax(,1)返回每列的最大值的索引，tf.equal(a,b)若a,b相同就返回True，不相同就返回False
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #計算數據中多少預測值是對的、多少是錯的
    #tf.cast(a,你要的型態) 變換型態
   
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    #預測值的準確度，這會是一個百分比
    return result

#define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784])#None: 不規定幾個Sample，784=每個輸入的大小
ys = tf.placeholder(tf.float32,[None,10])

prediction = add_layer(xs, 784, 10, activation_function = tf.nn.softmax)#softmax用在辨識
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                reduction_indices=[1])) #loss，詳情請見 https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E4%BB%8B%E7%B4%B9-%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8-loss-function-2dcac5ebb6cb
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)
    for i in range(1001):
        batch_xs, batch_ys = mnist.train.next_batch(100) #一次從Database拿100個東西出來
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
        if i %50 ==0:
            print("i= ", i, compute_accuarcy(
                mnist.test.images, mnist.test.labels
            ))



