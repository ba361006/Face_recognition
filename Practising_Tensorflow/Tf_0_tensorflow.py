import tensorflow as tf 


# a = tf.Variable(tf.random_normal([2,3]))
# b = tf.Variable(tf.random_normal([2,3]))

c = tf.constant([1,2,3])
d = tf.constant([[4,5,6],[7,8,9]])

ans1 = tf.reduce_sum(c*d)
ans2 = tf.reduce_sum(c*d,reduction_indices=[1])
ans3 = tf.reduce_mean(c*d,reduction_indices=[1])
ans4 = tf.reduce_mean(tf.reduce_sum(c*d,reduction_indices=[1]))
# ans3 = tf.matmul(c,d)

# init = tf.initialize_all_variables()


with tf.Session() as sess:
    # sess.run(init)
    print('a',sess.run(c))
    print('b',sess.run(d))
    print('ans1',sess.run(ans1))
    print('ans2',sess.run(ans2))
    print('ans3',sess.run(ans3))
    print('ans4',sess.run(ans4))



#%%
import tensorflow as tf 

cons1 = tf.constant([1,2,3,4,5])
cons2 = tf.constant([1,3,4,4,6])

eq1 = tf.equal(cons1,cons2)
cast_eq1 = tf.cast(tf.equal(cons1, cons2),tf.float32)

with tf.Session() as sess:
    print('cons1',sess.run(eq1))
    print('cast_eq1',sess.run(cast_eq1))
