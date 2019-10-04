import tensorflow as tf
import numpy as np

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])

product1 = tf.matmul(matrix1, matrix2) #martix multiply
product2 = np.dot(matrix1,matrix2)

#method 1
sess = tf.Session()
result = sess.run(product1)
print(result)
sess.close()

# #method 2
# with tf.Session() as sess: #with A as B : 打開A，以B命名，不用再關上sess 
#                            #因為他只在縮排內運行
#     result2 = sess.run(product1)
#     print(result2)
