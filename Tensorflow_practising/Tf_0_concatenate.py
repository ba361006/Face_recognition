import numpy as np 

a = [[1,2,3]]
b = [[4,5,6]]


c = [0,0,1]
d = [0,1,0]
xs = np.concatenate(a + b, axis = 0)
ys = np.concatenate((c, d), axis = 0)
print('xs:',xs)
print('ys:',ys)
print('xs.shape:',xs.shape)
print('ys.shape:',ys.shape)
print('type(xs):', type(xs))
