from urllib.request import urlretrieve
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt


def download():     # 把資料下載下來， 步驟: 創建資料夾-讀取資料-下載-分類
    categories = ['tiger', 'kittycat']
    for category in categories:
        # os.makedirs(path,parents=,exist_ok=): 創造資料夾
        # path: 要儲存的路徑, parents: 如果父目錄不存在，是否創建父目錄, exist_ok: 只有在目錄不存在時創建目錄，已存在也不會error
        os.makedirs('./For_transfer_learning/data/%s' % category, exist_ok=True)
        
        # open(file's name, mode[, buffering]): 開啟資料夾
        # name: 欲開啟文件的名字(包含路徑), mode: 開啟的模式 'r'為只讀, buffering為要不要暫存(先不用管)
        with open('./For_transfer_learning/imagenet_%s.txt' % category, 'r') as file: # 如果讀取的資料不見就會產生error可能不會關閉資料夾(占空間)所以一般會用with open as file，會自動file.close
            # file.readlines() 逐行讀取
            urls = file.readlines()
            n_urls = len(urls)
            
            # enumerate(sequence, [start=0]) 把sequence裡面的東西依照順序標齊  
            # sequence: 一個序列, start: 數字從幾開始，預設為0
            # # seasons = ['Spring', 'Summer', 'Fall', 'Winter'], 
            # # list(enumerate(seasons))
            # # [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
            for i, url in enumerate(urls): #依順序枚舉urls裡的資料並存放在url裡
                try:
                    # urlretrieve(url, filename=None, reporthook=None, data=None): 從遠端下載資料到本地
                    # url: 網址, filename: 指定本地儲存路徑, reporthook: 顯示下載進度, data: 返回一個包含兩個元素的元組(filename, headers) header表示服務器響應頭
                    
                    # strip([chars]): 移除頭尾指定的字串、預設為移除空格
                    
                    # split(str="", num=string.count(str)): 以指定的符號區隔整個字串且不包含指定的符號(空格也算字元)(在最後接[-1]表示只讀最後一個)
                    # 舉例: this is string wow以i為主→th, s is str,ng wow 
                    urlretrieve(url.strip(), './For_transfer_learning/data/%s/%s' % (category, url.strip().split('/')[-1]))
                    print('%s %i/%i' % (category, i, n_urls))
                except:
                    print('%s %i/%i' % (category, i, n_urls), 'no image')

################## 問題: Crop_img要幹嘛、resized返回值是甚麼 #################

def load_img(path): # 讀取圖片, /255壓縮並整理成[1, 224, 224, 3]的shape
    # skimage與opencv很像，處理圖片用的封包
    img = skimage.io.imread(path) # 讀取圖片
    img = img / 255.0 # 壓縮img的像素

    # print "Original Image Shape: ", img.shape
    # we crop image from center
    # min(img.shape[:2])是選擇img的0、1維其中比較小的的尺寸 [0]維(高度)、[1]維(寬度)
    short_edge = min(img.shape[:2]) 
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224, 3))[None, :, :, :]   # shape [1, 224, 224, 3]
    return resized_img

################# 問題: imgs[k].append(resized_img)在幹啥? 目前猜測resized_img返回值是一個圖片的資料，imgs[k]是把圖片的資料append到當前k(kittycat or tiger)的字典裡 ######################

def load_data(): #把所有Data整理成.jpg檔，偽造tigers_y, cat_y數據並傳回imgs['tiger'], imgs['kittycat'], tigers_y, cat_y
    imgs = {'tiger': [], 'kittycat': []}
    for k in imgs.keys(): # 返回imgs這個字典的目錄
        dir = './For_transfer_learning/data/' + k # dir = ./For_transfer_learning/data/tiger, ./For_transfer_learning/data/kittycat
        # os.listdir(path): 返回path裡所有資料夾的名字(包含副檔名)
        for file in os.listdir(dir): 
            
            # not: 用於判斷後面的式子是否為False，若是則True，否則False ；補充: None,  False, 空字符串"", 0, 空列表[], 空字典{}, 空元组()都相当于False
            # str.lower(): 把括號內的字變成小寫
            # str.endswith(suffix, start, end): 檢查字串是否以指定的字串結果，是就返回True，否則返回False
            # suffix: 指定的字串, start: 字串中的開始位置, end: 字串中的結束位置 
            if not file.lower().endswith('.jpg'): # 檢查是否結尾為jpg
                
                # continue是跳過迴圈下面的程式碼但是不跳出迴圈後重新開始for迴圈
                continue # 如果不是.jpg檔就跳過本次迴圈，也就是不載入imgs裡面
            try:
                # os.path.join(path1, path2): 把目錄和文件合併成一個路徑
                resized_img = load_img(os.path.join(dir, file)) 
            except OSError: # 如果try發生了OSError錯誤則continue, OSError: 路徑錯誤，例如檔案路徑'D:\LearningBooks\test.txt'，這裡\text的\t會被視為轉義符號，所以要就把\改成/，或是改為\\test.txt就會對了
                continue
            imgs[k].append(resized_img)    # [1, height, width, depth] * n
            if len(imgs[k]) == 400:        # only use 400 imgs to reduce my memory load
                break
    
    
    # fake length data for tiger and cat
    # 先用np.random.randn(len(imgs['tiger']), 1) 製造維度為(len(imgs['tiger']), 1)的亂數
    # 再用np.maximum(20, np.random.randn(len(imgs['tiger']), 1) * 30 + 100) 製造一組最低為20的亂數數據
    # np.maxium(x,y): x與y逐位比較大小，較大的印出來
    tigers_y = np.maximum(20, np.random.randn(len(imgs['tiger']), 1) * 30 + 100)
    cat_y = np.maximum(10, np.random.randn(len(imgs['kittycat']), 1) * 8 + 40)
    return imgs['tiger'], imgs['kittycat'], tigers_y, cat_y


class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68] # 一旦物件in被納入class vgg16則具備vgg_mean = [103.939, 116.779, 123.68]的屬性， 可用in.vgg_mean叫出

    def __init__(self, vgg16_npy_path=None, restore_from=None): # 一旦物件被納入class vgg16則具備__init__內的屬性且被納入的物件
        # pre-trained parameters
        try:
            # self.mathod 意思是一旦物件in執行def func的話會動態賦予物件.data_dict的屬性，但此處位於def __init__內，所以一旦物件被納入class Vgg16裡面就會自動有__init__裡面所有self.mathod的屬性
            # 沒有self.mathod的變數都屬於在def func 內的區域變數
                        
            # np.load(path,encoding): 用encoding的形式(utf-8...)從path讀取
            # .item(): 同時顯示字典的類別以及內容
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item() # 從vgg16_npy_path的位置讀取vgg16的參數並放入self.data_dict裡
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.tfy = tf.placeholder(tf.float32, [None, 1])

        # Convert RGB to BGR
        # tf.split(value, num_or_size_splits, axis=0, num=None, name='split')
        # value: 輸入, num_or_size_splits: 可以放tensor或是正整數(詳細看程式碼Tf_0_tf.split), axis=欲切割之維度 
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)   # self.tfx要*255.0在load_img時有/255壓縮過圖片

        # tf.concat(values, axis, name='concat'): t3, t4 with shapes [2, 3], tf.shape(tf.concat([t3, t4], 0))  # [4, 3] tf.shape(tf.concat([t3, t4], 1))  # [2, 6]
        # 此處blue, green, red的維度皆為[None, 224, 224, 1]， 把他們tf.concat之後就會變回原本的圖片bgr且維度為(blue, green, red)[None, 224, 224, 3]
        bgr = tf.concat(axis=3, values=[ 
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])
        #################################### 問題: -掉vgg_mean幹嘛? ###################################

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1') # output size = (-1, 112, 112, 3)

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2') # output size = (-1, 56, 56, 3)

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3') # output size = (-1, 28, 28, 3)

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4') # output size = (-1, 14, 14, 3)

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5') # output size = (-1, 7, 7, 3)
        # detach original VGG fc layers and reconstruct your own fc layers serve for your own purpose


       
        self.flatten = tf.reshape(pool5, [-1, 7*7*512]) 
        # tf.layers.dense(inputs, units, activation = None,....): 添加一個全連接層
        # inputs: 輸入, units: 輸出的維度
        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
        self.out = tf.layers.dense(self.fc6, 1, name='out')

        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph
            self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
            self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
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

    def predict(self, paths):
        fig, axs = plt.subplots(1, 2)
        for i, path in enumerate(paths):
            x = load_img(path)
            length = self.sess.run(self.out, {self.tfx: x})
            axs[i].imshow(x[0])
            axs[i].set_title('Len: %.1f cm' % length) 
            axs[i].set_xticks(()); axs[i].set_yticks(()) #隱藏x,y軸的值
        plt.show()

    def save(self, path='./For_transfer_learning/model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


def train():
    tigers_x, cats_x, tigers_y, cats_y = load_data()
    # print('tigers_x.shape:',tigers_x.shape) = list
    # print('tigers_y.shape:',tigers_y.shape) = (400,1)
    # print('cats_x.shape:',cats_x.shape) = list
    # print('cats_y.shape:',cats_y.shape) = (400,1)
    # print('type(tigers_x):', type(tigers_x)) = list
    # print('type(tigers_y):', type(tigers_y)) = numpy.ndarray
    # print('type(cats_x):', type(cats_x)) = list
    # print('type(cats_y):', type(cats_y)) = numpy.ndarray


    # plot fake length distribution
    plt.hist(tigers_y, bins=20, label='Tigers')
    plt.hist(cats_y, bins=10, label='Cats')
    plt.legend()
    plt.xlabel('length')
    plt.show()

    xs = np.concatenate(tigers_x + cats_x, axis=0)
    ys = np.concatenate((tigers_y, cats_y), axis=0)
    # xs.shape: (800, 224, 224, 3)
    # ys.shape: (800, 1)
    # type(xs,ys): numpy.array

    vgg = Vgg16(vgg16_npy_path='./For_transfer_learning/vgg16.npy')
    print('Net built')
    for i in range(100):
        # np.random.randint(low, high=None, size=None, dtype='l'): 隨機返回在最小值low 最大值high區間內的整數
        b_idx = np.random.randint(0, len(xs), 6)
        train_loss = vgg.train(xs[b_idx], ys[b_idx])
        print(i, 'train loss: ', train_loss)

    vgg.save('./For_transfer_learning/model/transfer_learn')      # save learned fc layers


def eval():
    vgg = Vgg16(vgg16_npy_path='./For_transfer_learning/vgg16.npy',
                restore_from='./For_transfer_learning/model/transfer_learn')
    vgg.predict(
        ['./For_transfer_learning/data/kittycat/000129037.jpg', './For_transfer_learning/data/tiger/391412.jpg'])


if __name__ == '__main__':
    # download()
    # train()
    eval()