import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
#placeholder(dtype,[列，行])
 
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]})) 
    #要用placeholder就要用feed_dict
    # Tensorflow 如果想要从外部传入data,
    # 那就需要用到 tf.placeholder(),
    