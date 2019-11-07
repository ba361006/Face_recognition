import tensorflow as tf

state = tf.Variable(0,name = 'counter') #tf.Variable是在Tensorflow中
                                        #創造一個變量，若沒用tf.Variable只是在
                                        #python中創造一個變量
print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one) #
update = tf.assign(state,new_value) # 在tensorflow中把new_value 丟給 state
                                    # 也就是一般程式概念中的 state = new_value

init = tf.initialize_all_variables() #如果在Tensorflow中有使用到變數就一定要初始化

with tf.Session() as sess:
    sess.run(init) #一定要先sess.run(init)
    for _ in range(3): # "_" 可以把它當成一個變量，也可以說我們只是想要這個地方跑三次，基本上"_"本身不重要
        sess.run(update)
        print(sess.run(state))