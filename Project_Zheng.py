from urllib.request import urlretrieve
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt


def download():     # download tiger and kittycat image
    categories = ['tiger', 'kittycat']
    for category in categories:
        os.makedirs('./For_transfer_learning/data/%s' % category, exist_ok=True)
        with open('./For_transfer_learning/imagenet_%s.txt' % category, 'r') as file:
            urls = file.readlines()
            n_urls = len(urls)
            for i, url in enumerate(urls):
                try:
                    urlretrieve(url.strip(), './For_transfer_learning/data/%s/%s' % (category, url.strip().split('/')[-1]))
                    print('%s %i/%i' % (category, i, n_urls))
                except:
                    print('%s %i/%i' % (category, i, n_urls), 'no image')


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


def load_data(): # 載入圖片時，把label與圖片的資料存到字典"imgs"裡，label起始值為1
    imgs = {'no_1': [],
    'no_2': [],
    'no_3': [],
    'no_4': [],
    'no_5': [],
    'no_6': [],
    'no_7': [],
    'no_8': [],
    'no_9': [],
    'no_10': []
    }
    count = 0

    for  k in imgs.keys(): 
        labels = np.zeros([len(imgs)]) # imgs內有幾個種類就先創造多少個label
        labels[count] = 1 
        dir = './For_transfer_learning/data/' + k
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
    return imgs['no_1'], imgs['no_2'], imgs['no_3'], imgs['no_4'], imgs['no_5'], imgs['no_6'], imgs['no_7'], imgs['no_8'], imgs['no_9'], imgs['no_10']


def label_split(inputs):
    label = []
    imgs = []
    for i in range(0, len(inputs)):
        indices, resized_img = inputs[i]
        label.append(indices)
        imgs.append(resized_img)
    return label,imgs



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
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
        self.out = tf.layers.dense(self.fc6, 10, name='out') # 種類變多這裡要改

        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph
            self.prediction = tf.nn.softmax(self.out)
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.tfy * tf.log(self.prediction)))
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
            # self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
            # self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
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
        label = labels[0]
        test_x = load_img(paths) #np.ndarray
        test_y = np.zeros((1,labels[1]))
        test_y[0,label] = 1 # type(test_y): ndarray

        pre_x = self.sess.run(self.out, feed_dict = {self.tfx: test_x})
        print('pre_x:' ,pre_x)

        print('tf.argmax(pre_x)):', self.sess.run(tf.argmax(pre_x,1)))
        print('tf.argmax(test_y):', self.sess.run(tf.argmax(test_y,1)))
        correct_predcition = tf.equal(tf.argmax(pre_x,1), tf.argmax(test_y,1))# type(tf.argmax(pre_x,1)): ndarray
        print('correct_prediction:', self.sess.run(correct_predcition))
        
        accuracy = tf.reduce_mean(tf.cast(correct_predcition, tf.float32))
        print('accuracy:', self.sess.run(accuracy))

        result = self.sess.run(accuracy, feed_dict = {self.tfx: test_x})
        print('result: ', result)
        return(result)


    def save(self, path='./For_transfer_learning/model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


def train():
    no_1, no_2, no_3, no_4, no_5, no_6, no_7, no_8, no_9, no_10 = load_data()
    # print('no_1:', type(no_1[0][0]),type(no_1[0][1]))

    label_1, no1_x = label_split(no_1)
    label_2, no2_x = label_split(no_2)
    label_3, no3_x = label_split(no_3)
    label_4, no4_x = label_split(no_4)
    label_5, no5_x = label_split(no_5)
    label_6, no6_x = label_split(no_6)
    label_7, no7_x = label_split(no_7)
    label_8, no8_x = label_split(no_8)
    label_9, no9_x = label_split(no_9)
    label_10, no10_x = label_split(no_10)

    xs = np.concatenate(no1_x + no2_x + no3_x + no4_x + no5_x + no6_x + no7_x + no8_x + no9_x + no10_x, axis = 0)
    ys = np.concatenate((label_1, label_2, label_2, label_2, label_3, label_4, label_5, label_6, label_7, label_8, label_9, label_10), axis = 0)

    vgg = Vgg16(vgg16_npy_path='./For_transfer_learning/vgg16.npy')
    print('Net built')
    
    for i in range(1001):
        b_idx = np.random.randint(0, len(xs), 6)
        train_loss = vgg.train(xs[b_idx], ys[b_idx])
        if i % 100 == 0:
            print(i, 'train loss: ', train_loss)

    vgg.save('./For_transfer_learning/model/transfer_learn')      # save learned fc layers


def eval():
    vgg = Vgg16(vgg16_npy_path='./For_transfer_learning/vgg16.npy',
                restore_from='./For_transfer_learning/model/transfer_learn')
    vgg.predict(
        './For_transfer_learning/data/test_no_1/0506_01.jpg', labels = [4,10]) # labels[x,y], x:第幾種，從零開始, y總共幾種


if __name__ == '__main__':
    # download()
    train()
    # eval()