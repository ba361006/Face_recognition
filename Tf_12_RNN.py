import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

#hyperparameters
lr = 0.001 # 學習率
training_iters = 100000 # 學習次數
batch_size = 128 # 有幾筆資料 隨便打的

n_inputs = 28 # MNIST data inputs(img shape:28*28) 圖片的columns
n_steps = 28 # time steps 圖片的row
n_hidden_units = 128 # neurons in hidden layer 亂設的
n_classes = 10 # MNIST classes (0-9 digits)

x = tf.placeholder(tf.float32,[None, n_steps, n_inputs])
y = tf.placeholder(tf.float32,[None, n_classes])


# Define weights
weights = {
    # (28,128)
    'in':tf.Variable(tf.random_normal([n_inputs, n_hidden_units])), 

    # (128,10)
    'out':tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}


biases = {
    # (128,)
    'in':tf.Variable(tf.constant(0.1, shape = [n_hidden_units,])),

    # (10,)
    'out':tf.Variable(tf.constant(0.1, shape = [n_classes, ]))
}

def RNN(X, weights, biases):
    # hidden layer for input to cell
    ###############################################
    # X (128 batch, 28 steps, 28 inputs)
    
    X = tf.reshape(X, [-1, n_inputs]) # 轉換成(128*28, 28 inputs) 
    X_in = tf.matmul(X, weights['in']) + biases['in'] # 轉換成(128 batch * 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units]) # 轉換成(128 batch , 28 steps, 128 hidden)



    #cell
    ###############################################
    # tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # n_hidden: 神經元的個數
    # forget_bias: LSTM的忘記係數，若為1則不會忘記；0則都忘記
    # state_is_tuple: 建議用True；裡面會存在一個初始狀態函數zero_state(batch_size, dtype)，可以用lstm_cell.zero_state取出來
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias = 1.0, state_is_tuple = True)
    # lstm cell is divided into two parts (c_state, m_state)

    # 取出零狀態
    # .zero_state(batch_szie, state_size)
    # zero_state: batch_size是輸入樣本批次的數目；dtype是資料型態
    __init__state = lstm_cell.zero_state(batch_size, dtype = tf.float32) 


    # cell的運作方式就是對他輸入前一個狀態h0、輸入in_0，他就會吐出來一個狀態h1、輸出out_0，再輸入h1、in_1就會得到h2、in_2.....，而tf.nn.dynamic_rnn就是再縮短這個重複呼叫cell的方法
    # tf.nn.dynamic_rnn(cell, inputs, initial_state, time_major)
    # inputs: 輸入圖片，★重點★ 輸入的型態為(batch_size, time_steps, input_size)，其中的time_steps有點像是
    # initial_state: 初始的狀態（由.zero_state得到）
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state = __init__state, time_major = False)
    # outputs: 是一個list
    # states: (batch_size, cell.state_size)


    #hidden layer for output as the final results
    ###############################################
    results = tf.matmul(states[1], weights['out']) + biases['out']#[1] => m_state
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train], feed_dict = { x:batch_xs, y:batch_ys})
        if step % 20 == 0:
            print('steps:',step, sess.run(accuracy, feed_dict = {x:batch_xs, y:batch_ys}))
        step += 1
