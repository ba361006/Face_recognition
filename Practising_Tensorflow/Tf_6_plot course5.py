#題目是預測y = x^2 -0.5 + noise，x的大小為(300,1)300行、一列
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-1,1,300)[:, np.newaxis]#創造一個linspace格式是一行很多列
noise = np.random.normal(0,0.05,x_data.shape)#雜訊的格式跟x_data一樣
y_data = np.square(x_data) - 0.5 + noise 


xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])


def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))#生成初始位置時，隨機變量比全0較好
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) #zeros(幾列,幾行)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases #Tensorflow可以矩陣加向量，但是矩陣的Colum數跟向量的元素數要一致，所以在這裡兩個的Colum數都是output_size
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
                     reduction_indices=[1])) #reduction_indices[0] = 把行壓扁便把行的值都加起來，[1]則是壓扁列，也可以打成[0,1]就是先壓扁行再壓扁列
#loss取有效值(均方根)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#用梯度法以0.1的學習率對loss進行更正or提升，讓下次會有更好的結果

#================================== 建構神經層 ===================================

fig = plt.figure() #創造一個圖片框
ax = fig.add_subplot(1,1,1)  #連續畫圖，與plt.subplot(111)相同功能，但不知道兩個有甚麼差
ax.scatter(x_data, y_data)   #把數據用點的形式plot出來
plt.ion()  # plot完不要暫停繼續做(interaction on)，interaction on之後就可以跟圖互動，所以新訓練完的值會再一次被plt.show上去
plt.show() # show圖，只能把圖SHOW出來後就暫停整個程序，就像是算完畫完圖之後交交出去就不能改了一樣，會把算到這裡的值印上去之後就不會修改了

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1001):
        sess.run(train_step, feed_dict = {xs: x_data, ys: y_data})
        if i % 50 ==0:
            print("i= ",i," ",sess.run(loss,feed_dict={xs: x_data, ys: y_data}))
            
            try: #try與except的語句意思就是先執行try下的語句若有誤則執行except下的語句；若try下的語句無誤則不會執行except下的語句且程式繼續往下走
                 #詳細內容在 https://pydoing.blogspot.com/2011/01/python-try.html
                
                ax.lines.remove(lines[0]) #去除掉第一條線
            
            except Exception: #Exception是except的名字
                pass #啥都不做

            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            #單求x的預測值
            
            lines = ax.plot(x_data, prediction_value,'r-',lw=5)
            #跟imshow很像，把x,y的值丟進去，用紅色的線'r-'，線寬是5 lw=5
            plt.pause(0.1)
            #每做一次就暫停0.1秒 不然太快會看不出來




