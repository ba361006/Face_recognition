import numpy as np 
b = [1]
a = np.array([0,0])

c = [b,a]
print('c[0]', c[0])
print('c[1]', c[1])
print(type(c[0]),type(c[1]))
# imgs = {'no_1': [], 'no_2': []}
# count = 0

# for  k in imgs.keys():
#     print(k)
#     label = np.array([0,0])
#     label[count] = 1
#     count += 1
#     print('label:',label)
#     print('count:',count)