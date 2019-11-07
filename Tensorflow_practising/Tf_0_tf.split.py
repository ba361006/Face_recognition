import numpy as np
import tensorflow as tf  
vgg_mean = np.reshape(range(150),(5,30))

# tf.split(value, num_or_size_splits, axis=0, num=None, name='split')
a, b, c, x, y = tf.split(vgg_mean, 5, 1)    # num_or_size_splits若為整數則需要能與vgg_mean.shape[1]整除 (因為此範例axis=1)，且會平均分配到a b c x f 上
d, e, f, w, r = tf.split(vgg_mean, [3, 4, 5, 8, 10], 1) # [3 4 5 8 10] 需加起來能與vgg_mean.shape[1]一樣Q (因為此範例axis=1)，且會把切割出來的維度對應到d e f w r 上

with tf.Session() as sess:
    print('vgg.shape:',vgg_mean.shape)
    print('vgg.shape[0]:',vgg_mean.shape[0]) 
    print('vgg.shape[1]:',vgg_mean.shape[1])    # 此處注意vgg_mean.shape[1] = 30 是由高維度往低維度算!!
    print('len:', len(vgg_mean))


    print('tf.split_integer')
    print(sess.run(tf.shape(a)))
    print(sess.run(tf.shape(b)))
    print(sess.run(tf.shape(c)))
    print(sess.run(tf.shape(x)))
    print(sess.run(tf.shape(y)))


    print('tf.split_list')
    print(sess.run(tf.shape(d)))
    print(sess.run(tf.shape(e)))
    print(sess.run(tf.shape(f)))
    print(sess.run(tf.shape(w)))
    print(sess.run(tf.shape(r)))