import numpy as np 

def imgs():
    imgs = {'no_1': [(1,'i2'),(1,'i4'),(1,'i6'),(1,'i8')], 'no_2': [(2,'q2'),(2,'q4'),(2,'q6'),(2,'q8')]}
    return imgs['no_1'], imgs['no_2']

def label_split(inputs):
    label = []
    imgs = []
    for i in range(0, len(inputs)):
        indices, resized_img = inputs[i]
        label.append(indices)
        imgs.append(resized_img)
    return label,imgs

q,w = imgs()
label_q, imgs_q = label_split(q)
label_w, imgs_w = label_split(w)
print('label_q: %s imgs_q: %s' % (label_q , imgs_q))
print('label_w: %s imgs_w: %s' % (label_w , imgs_w))

xs = np.concatenate(imgs_q + imgs_w, axis = 0)
# print('len:',len(out))
# print('out:',out[:][1])
# print('type:',type(out))
# print('shape:',out.shape)