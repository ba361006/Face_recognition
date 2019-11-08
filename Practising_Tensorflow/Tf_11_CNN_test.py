import tensorflow as tf  
      
a = tf.constant([  
            [[1.0,2.0,3.0,4.0],  
            [5.0,6.0,7.0,8.0],  
            [8.0,7.0,6.0,5.0],  
            [4.0,3.0,2.0,1.0]],  
            [[4.0,3.0,2.0,1.0],  
             [8.0,7.0,6.0,5.0],  
             [1.0,2.0,3.0,4.0],  
             [5.0,6.0,7.0,8.0]]  
        ])  
b = tf.reshape(a,[1,4,4,2])  
      
pooling=tf.nn.max_pool(b,[1,2,2,1],[1,1,1,1],padding='VALID')  

with tf.Session() as sess:  
    print('a:', sess.run(a))
    print('b:', sess.run(b)) 
    print("reslut:")  
    result=sess.run(pooling)  
    print (result)  