import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import collections 
import random


category = 10 # 種類
train_file = 'no_' # 訓練資料夾名稱，此處名稱要與測試資料的實際資料夾名稱相同，且從0開始排序，例如no_0, no_1, no_2 ....
test_path = './For_transfer_learning/data/test_no_0' # 測試資料夾路徑、要加雙引號
test_label = 0 # 測試資料的正確標籤




def load_data(): # 載入圖片時，把label與圖片的資料存到字典"imgs"裡，label起始值為1
    print('\n ### loading data ###')

    imgs = build_dict()# 種類
    count = 0

    for  k in imgs.keys(): 
        labels = np.zeros([1,len(imgs)]) # imgs內有幾個種類就先創造多少個label
        labels[0][count] = 1 
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
    # 每個imgs['no_x']裡面的資料都是(label, resized_img), label: [?,0,0,0,0,0,0,0,0,0]
    return imgs

def build_dict():# 建立字典 
    dict_name = collections.OrderedDict()

    for i in range(category):
        contents = train_file + str(i) #資料夾名稱
        dict_name[contents] = []
    return dict_name


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
    resized_img = skimage.transform.resize(crop_img, (224, 224, 3))[None, :, :, :]   # shape [1, 224, 224, 3]只能測試33張 # 圖片大小 112 112 3 → 4*4*512
    return resized_img




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

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])# 圖片大小 112 112 3 → 4*4*512
        self.tfy = tf.placeholder(tf.float32, [None, category])# 種類

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
        self.flatten = tf.reshape(pool5, [-1, 7*7*512]) # self.flatten.shape (?, 25088)# 圖片大小 112 112 3 → 4*4*512 ; 224 224 3 → 7*7*512

        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
        self.out = tf.layers.dense(self.fc6, category, name='out') #  種類
        self.test_out = tf.nn.softmax(self.out)


        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.out, labels = self.tfy ))
            self.train_op = tf.train.AdamOptimizer(1e-6).minimize(self.loss)
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

        print('\n### predicting ###')


        pre_x = self.sess.run(self.test_out, feed_dict = {self.tfx: test_x})
        print('pre_x:' ,pre_x)# 印出預測每個label的機率 np.ndarray(?, 10) 
        print('tf.argmax(pre_x)):', self.sess.run(tf.argmax(pre_x,1))) # 印出模型預測的label
        print('tf.argmax(test_y):', self.sess.run(tf.argmax(test_y,1)))# 印出實際輸入的label


        correct_predcition = tf.equal(tf.argmax(pre_x,1), tf.argmax(test_y,1))# type(tf.argmax(pre_x,1)): ndarray
        print('correct_prediction:', self.sess.run(correct_predcition))# 印出模型預測與實際輸入的比對結果
        

        accuracy = tf.reduce_mean(tf.cast(correct_predcition, tf.float32))
        print('\naccuracy rate:', self.sess.run(accuracy))# 印出正確率



    def save(self, path='./For_transfer_learning/model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


def train():
    imgs = load_data()
    vgg = Vgg16(vgg16_npy_path='./For_transfer_learning/vgg16.npy')
    print('Net built')
    
    
    for k in range(10001): # 1001 0.00082637486
        count = 0
        for i in range(10):
            random_idx= random.randint(0,99)
            content = train_file + str(count)# 資料夾名稱
            train_loss = vgg.train(imgs[content][random_idx][1], imgs[content][random_idx][0])
            # imgs[content][random_idx][1].shape = np.ndarray(10,224,224,3)] (image)
            # imgs[content][random_idx][0].shape = np.ndarray(10,10) label
            count += 1
        if k % 100 == 0:
            print(k, 'train loss: ', train_loss)


    vgg.save('./For_transfer_learning/model/transfer_learn')      # save learned fc layers


def eval():
    print('\n### test_label: %s ###' % (test_label))
    vgg = Vgg16(vgg16_npy_path='./For_transfer_learning/vgg16.npy',
                restore_from='./For_transfer_learning/model/transfer_learn')
    

    
    vgg.predict(
        test_path, labels = [test_label,category]) # 種類
        # labels[x,y], x: 第0~9號, y: 總共幾種


if __name__ == '__main__':
    # train()
    eval()

# 訓練資料問題導致正確率下降
# batch: 1, (?,224,224,3), 1e-6  訓練至4400時loss = 0 
# 總結來說效果還不錯，除了label 0 本身測試資料正臉照片較少導致正確率只有36%以外， 其他label 1 2 都有 85%左右的正確率