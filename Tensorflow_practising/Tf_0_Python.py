import numpy as np


#numpy是一種在python中常常用到的數學模組，自己上網查是甚麼

#================================== Python資料型態練習 ===================================
a = np.random.rand(10,2)
print('a:',a)
print('a.shape:',a.shape)
print('type(a)',type(a))

b = np.array([[1],[2],[3]])
print('b:',b)
print('b.shape:',b.shape)
print('type(b)',type(b))


x_data = np.linspace(-1,1,3)[:,np.newaxis]
print('x_data:',x_data)
print('x_data.shape:',x_data.shape)
print('type(x_data',type(x_data))

################################################################################

c = np.array([[3,4,5],[4,5,6]])
print('c:',c)

d = np.array([[1,2],[2,3],[3,4]])
print('d:',d)

c_d = np.matmul(c,d)
print('c_d:',c_d)

d_c = np.matmul(d,c)
print('d_c:',d_c)

#================================== Python資料型態練習 ===================================