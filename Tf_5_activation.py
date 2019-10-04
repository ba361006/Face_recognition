import tensorflow as tf
import numpy as np


#================================== Python資料型態練習 ===================================
# a = np.random.rand(10,2)
# print('a:',a)
# print('a.shape:',a.shape)
# print('type(a)',type(a))

# b = np.array([[1],[2],[3]])
# print('b:',b)
# print('b.shape:',b.shape)
# print('type(b)',type(b))


# x_data = np.linspace(-1,1,3)[:,np.newaxis]
# print('x_data:',x_data)
# print('x_data.shape:',x_data.shape)
# print('type(x_data',type(x_data))

################################################################################

# b = np.array([[1,2],[2,3],[3,4]])
# print('b:',b)

# c = np.array([[3,4,5],[4,5,6]])
# print('c:',c)

# b_c = np.matmul(b,c)
# print('b_c:',b_c)

# c_b = np.matmul(c,b)
# print('c_b:',c_b)

#================================== Python資料型態練習 ===================================



x_data = np.linspace(-1,1,300)[:, np.newaxis]#創造一個linspace格式是一行很多列
noise = np.random.normal(0,0.05,x_data.shape)#雜訊的格式跟x_data一樣
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])


def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))#生成初始位置時，隨機變量比全0較好
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) #zeros(幾列,幾行)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


#================================== 建構神經層 ===================================

#假設我們想建立一個三層神經網路，分別是輸入層，隱藏層，輸出層
#輸入層只有一個屬性就是x_data，所以輸入層只有一個神經元
#隱藏層假設我們想要10個神經元，而輸出層y_data只有一個屬性，所以也是一個神經元


l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
#格式: add_layer(inputs,in_size,out_size,activation_function=None)
#l1為輸入層，in_size為這層的神經元數，out_size為要輸出給下一層的神經元數

prediction = add_layer(l1,10,1,activation_function=None)
#假設隱藏層沒有激勵函數


loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))
#loss取有效值(均方根)
#reduction_indices介紹: https://blog.csdn.net/qq_33096883/article/details/77479766

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#用梯度法以0.1的學習率對loss進行更正or提升，讓下次會有更好的結果

#================================== 建構神經層 ===================================

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print("prediction = ",prediction)
    for i in range(1001):
        sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
        if i % 50 ==0:
            print("i= ",i," ",sess.run(loss,feed_dict={xs: x_data, ys: y_data}))





