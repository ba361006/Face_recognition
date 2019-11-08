import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3


#######################    建構Tensprflow的架構    #######################
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) #Weights & biases 要是Variable 因為他們會在下面被訓練，訓練後要能把值丟回來在繼續跑所以要是變數
biases = tf.Variable(tf.zeros([1]))                      #若是常數就不能訓練了，有點像是在算遞迴函數那樣


y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data)) #loss的方式是均方差
optimizer =  tf.train.GradientDescentOptimizer(0.5) #用梯度下降法
train = optimizer.minimize(loss) #一般用在要訓練的目標函數，通常是均方差

init = tf.initialize_all_variables() #初始化變數

########################    建構Tensprflow的架構    #######################


sess = tf.Session()
sess.run(init)  #Very important

for step in range(201): #訓練兩百零一次 (計算到201，但不顯示201，也就是0~200共201個數字)
    sess.run(train) #開始訓練
    if step % 20 ==0: #每二十步把值抓出來看訓練到哪裡了
        print(step,sess.run(Weights),sess.run(biases)) #把值印出來，Weights跟biases的前面一定要打sess.run才可以看出來他們在Tensorflow運行中的變化


#25~31行也可以合併成
# with tf.Session() as sess:
#         sess.run(init)
#         for step in range(201):
#                 sess.run(train)
#                 if step % 20 == 0:
#                         print(step, sess.run(Weights), sess.run(biases))