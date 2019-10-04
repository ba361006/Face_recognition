import numpy as np 

a = [[1,2,3]]
b = [[4,5,6]]

xs = np.concatenate(a + b, axis = 0)
ys = np.concatenate((a, b), axis = 0)
print('xs:',xs)
print('ys:',ys)