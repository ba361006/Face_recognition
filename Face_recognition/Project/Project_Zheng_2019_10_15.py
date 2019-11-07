from urllib.request import urlretrieve
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import collections 




def load_data(): # 載入圖片時，把label與圖片的資料存到字典"imgs"裡，label起始值為1
    imgs = collections.OrderedDict()
    imgs['no_0'] = []
    imgs['no_1'] = []
    imgs['no_2'] = []
    imgs['no_3'] = []
    imgs['no_4'] = []
    imgs['no_5'] = []
    imgs['no_6'] = []
    imgs['no_7'] = []
    imgs['no_8'] = []
    imgs['no_9'] = []
    count = 0

    for  k in imgs.keys(): 
        labels = np.zeros([len(imgs)]) # imgs內有幾個種類就先創造多少個label
        labels[count] = 1 
        dir = './For_transfer_learning/data/' + k
        print('imgs.keys(): ', k)
        for file in os.listdir(dir):
            if not file.lower().endswith('.jpg'):
                continue
            try:
                resized_img = load_img(os.path.join(dir, file))
            except OSError:
                continue
            
            imgs[k].append((labels ,resized_img))    # [1, height, width, depth] * n
            if len(imgs[k]) == 100:        # only use 100 imgs to reduce my memory load
                break
        count += 1
    # 每個imgs['no_x']裡面的資料都是(label, resized_img)，共100個, label: [?,0,0,0,0,0,0,0,0,0]
    return imgs['no_0'], imgs['no_1'], imgs['no_2'], imgs['no_3'], imgs['no_4'], imgs['no_5'], imgs['no_6'], imgs['no_7'], imgs['no_8'], imgs['no_9']


def extract_data(dict_in):
    k = 0
    for i in range(10):
        content_name = 'no_' + str(i) # 創造no_?
        content_data = dict_in[content_name] # 讀取字典目錄內的資料 content = imgs['no_?']
        # print('content: %s, %s' %(content_name,content_data))
        label, data = label_split(content_data)# 把資料裡面的label與data分開

        if k == 0:#這裡用if單純只是如果直接打else內的東西會錯
            xs = data
            ys = label
            k = 1
        else:
            xs = np.concatenate((xs , data), axis = 0)
            ys = np.concatenate((ys, label), axis = 0)
    return xs, ys 


def load_img(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224, 3))[None, :, :, :]   # shape [1, 224, 224, 3]
    return resized_img


def label_split(inputs):# 輸入: (label, imgs)， 輸出: 分割輸入資料的labels ,imgs，按照輸入順序排列且type為list
    labels = []
    imgs = []
    for i in range(len(inputs)):
        indices, resized_img = inputs[i]
        labels.append(indices)
        imgs.append(resized_img)
    return labels, imgs # 返回labels = [label1, lable2, ...], imgs = [resized_img1, resized_img2, ....] 都是list



def file2img(input):# 輸入: 資料夾路徑， 輸出: 該資料夾內所有檔案路徑，type為np.ndarray
    i = 0 
    for file in os.listdir(input):
        img_path = os.path.join(input, file)
        if i == 0:
            output = load_img(img_path)
            i = 1
        else:
            output = np.concatenate((output, load_img(img_path)), axis = 0)
    print('\n shape of test_data:', output.shape) 
    
    return output






class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.tfy = tf.placeholder(tf.float32, [None, 10])# 種類變多這裡要改

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])


        
        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')


        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')


        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')


        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')


        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')


        # detach original VGG fc layers and
        # reconstruct your own fc layers serve for your own purpose
        self.flatten = tf.reshape(pool5, [-1, 7*7*512]) # self.flatten.shape (?, 25088)

        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
        self.out = tf.layers.dense(self.fc6, 10, name='out') # 種類變多這裡要改
        self.test_out = tf.nn.softmax(self.out)


        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph
            # self.prediction = tf.nn.softmax(self.out)
            # self.loss = tf.reduce_mean(-tf.reduce_sum(self.tfy * tf.log(self.prediction)))

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.out, labels = self.tfy ))
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
            # self.train_op = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x, self.tfy: y})

        return loss

    def predict(self, paths, labels):#paths, labels = list
        # 取得並處理測試資料的labels
        label = labels[0]# 取得實際輸入的label
        test_y = np.zeros((1,labels[1]))# 創造一排大小為(1, 總label數)的全0矩陣
        test_y[0,label] = 1 # type(test_y): ndarray
        
        # 取得測試資料的路徑
        test_x = file2img(paths)# type(test_x) <class 'numpy.ndarray'>,  test_x.shape (資料夾內照片的數量, 224, 224, 3)

        pre_x = self.sess.run(self.test_out, feed_dict = {self.tfx: test_x})
        # print('pre_x:' ,pre_x)# 印出預測每個label的機率 np.ndarray(?, 10)
        print('tf.argmax(pre_x)):', self.sess.run(tf.argmax(pre_x,1))) # 印出模型預測的label
        print('tf.argmax(test_y):', self.sess.run(tf.argmax(test_y,1)))# 印出實際輸入的label

        correct_predcition = tf.equal(tf.argmax(pre_x,1), tf.argmax(test_y,1))# type(tf.argmax(pre_x,1)): ndarray
        # print('correct_prediction:', self.sess.run(correct_predcition))# 印出模型預測與實際輸入的比對結果
        
        accuracy = tf.reduce_mean(tf.cast(correct_predcition, tf.float32))
        print('\n')
        print('accuracy rate:', self.sess.run(accuracy))# 印出正確率



    def save(self, path='./For_transfer_learning/model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


def train():
    no_0, no_1, no_2, no_3, no_4, no_5, no_6, no_7, no_8, no_9 = load_data()


    label_0, no0_x = label_split(no_0)
    label_1, no1_x = label_split(no_1)
    label_2, no2_x = label_split(no_2)
    label_3, no3_x = label_split(no_3)
    label_4, no4_x = label_split(no_4)
    label_5, no5_x = label_split(no_5)
    label_6, no6_x = label_split(no_6)
    label_7, no7_x = label_split(no_7)
    label_8, no8_x = label_split(no_8)
    label_9, no9_x = label_split(no_9)
    #label, no 都是list

    #xs, ys 都是np.ndarray
    xs = np.concatenate(no0_x + no1_x + no2_x + no3_x + no4_x + no5_x + no6_x + no7_x + no8_x + no9_x, axis = 0)
    # 會變成(resized_img1, resized_img2, ...),shape:(1000,224,224,3)
    
    ys = np.concatenate((label_0, label_1, label_2, label_2, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9), axis = 0)
    # 會變成([0,0,0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0]...),shape:(1000,10)
    
    vgg = Vgg16(vgg16_npy_path='./For_transfer_learning/vgg16.npy')
    print('Net built')
    
    for i in range(1001):
        b_idx = np.random.randint(0, len(xs), 10)
        train_loss = vgg.train(xs[b_idx], ys[b_idx])# xs[b_idx].shape = (10,224,224,3) ; ys[a_idx].shape = (10,10)
        if i % 100 == 0:
            print(i, 'train loss: ', train_loss)


    vgg.save('./For_transfer_learning/model/transfer_learn')      # save learned fc layers


def eval():
    vgg = Vgg16(vgg16_npy_path='./For_transfer_learning/vgg16.npy',
                restore_from='./For_transfer_learning/model/transfer_learn')

    # labels[x,y], x: 第0~9號, y: 總共幾種
    vgg.predict(
        './For_transfer_learning/data/test_no_0', labels = [0,10]) 
        


if __name__ == '__main__':
    # train()
    eval()