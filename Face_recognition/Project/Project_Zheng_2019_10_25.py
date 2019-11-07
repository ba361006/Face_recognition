from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import collections 
import cv2
import time 
import shutil

# train_variable
category = 10 # 種類
train_file = 'no_' # 訓練資料夾名稱，此處名稱要與測試資料的實際資料夾名稱相同，且從0開始排序，例如no_0, no_1, no_2 ....
train_cap_path = './For_transfer_learning/data/no_0/'# 訓練資料

# test_variable
test_path = './For_transfer_learning/data/test_no_2/' # 測試資料夾路徑、要加雙引號
test_label = 2 # 測試資料的正確標籤
test_cap_path = './For_transfer_learning/data/test_no_0/'# 測試資料

force_in = './For_transfer_learning/data/reinforcement/reinforcement_in/'



def load_data(): # 載入圖片時，把label與圖片的資料存到字典"imgs"裡，label起始值為1

    dict_imgs = build_dict(train_file, category)# 種類
    dict_labels = build_dict(train_file, category)# 
    count = 0

    for  k in dict_imgs.keys(): 
        labels = np.zeros([1,len(dict_imgs)]) # imgs內有幾個種類就先創造多少個label
        labels[0][count] = 1 
        dir = './For_transfer_learning/data/' + k

        print('\n### Loading %s ###\n' % (k))

        for file in os.listdir(dir):
            if not file.lower().endswith('.jpg'):
                continue
            try:
                resized_img = load_img(os.path.join(dir, file))
            except OSError:
                continue
            
            dict_imgs[k].append(resized_img)    # [1, height, width, depth] * n
            dict_labels[k].append(labels) 
            if len(dict_imgs[k]) == 100:        # only use 100 imgs to reduce my memory load
                break
        count += 1
    # 每個imgs['no_x']裡面的資料都是(label, resized_img), label: [?,0,0,0,0,0,0,0,0,0]
    return dict_imgs, dict_labels

def build_dict(content_name,category_number):# 建立字典 
    dict_name = collections.OrderedDict()

    for i in range(category_number):
        contents = content_name + str(i) #資料夾名稱
        dict_name[contents] = []
    return dict_name


def load_img(path):
    img = skimage.io.imread(path)
    img = img / 255.0
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
    
    return output


def get_train_data(input_path):
    folder_path = os.path.exists(input_path)
    folder_name = os.path.basename(os.path.normpath(input_path))# 取得input_path資料夾路徑的最後一段名稱 './For_transfer_learning/data/test_no_3/' → test_no_3

    if folder_path:
        shutil.rmtree(input_path)# 刪除原資料夾(為了清空舊資料)
    os.makedirs(input_path)# 創造新資料夾

    get_test_data(force_in, waitkey = 25, number = 40)#透過攝影機拿到資料，因為此處的程式並不會顯示在主程式，這個函數名稱就為了只是為了在主程式好看而已
    reinforcement(force_in, input_path, folder_name, 5)



def reinforcement(input_path , output_path, output_name, number):

    datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,)

    print('\n### reinforcing pictures from: %s ###\n' %(input_path))
    k = 0
    for file in os.listdir(input_path):
        k += 1
        path = input_path + file
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((1,)+img.shape)
        i = 0
        for batch in datagen.flow(img, batch_size = 2, save_to_dir = output_path, save_prefix = output_name, save_format = 'jpg'):
            i += 1
            if i == number:# 增強幾張
                break
    
    print('\n### %s pictures have been reinforced into %s pictures ####\n'% (k, (k * number)) )



def get_test_data(output_path, waitkey, number): # 每0.1秒拍一張人臉的照片，存到"./For_transfer_learning/data/zheng/before_reinforcement"裡面
    num = 0
    cap = cv2.VideoCapture(0)

    width = cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)

    while(True):
        faceCascade = cv2.CascadeClassifier('D:/Tools/Anaconda/envs/tensorflow/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2, minNeighbors = 4, minSize = (30,30))
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(
                frame,
                
                (x - 10,y - 10),
                (x + 10 + w,y + 10 + h),
                (0,255,0),
                
            )
        cv2.imshow('frame',frame)

        
        try:# 想要改成中間如果人臉消失了他要停在原地，但現在是人臉如果消失他會繼續拍
            if cv2.waitKey(waitkey): 
                img = frame[y:y+h,x:x+w]
                cv2.imwrite(output_path + 'face_' + str(num) + '.jpg', img) #按照儲存順序命名檔案
                print('\n### Taking %s picture ###' %(num))
                num += 1 #這裡是利用字串的方式對'face'加上'數字'以及'jpg'
                if num == number:
                    print('\n### Finishing Capturing ###\n')
                    break
        except:
            continue

    cap.release()
    cv2.destroyAllWindows()







class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('\n### Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr ###\n')

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
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
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
        print('\n### test_label: %s ,shape of test_data: %s ###' % (test_label,test_x.shape,)) # 如果要利用"%s"印出tuple，像是.shape這種東西的話要多寫一個',' 例如 '... %s' % (xxxx,)
        print('\n### predicting ###\n')


        pre_x = self.sess.run(self.test_out, feed_dict = {self.tfx: test_x})
        # print('pre_x:' ,pre_x)# 印出預測每個label的機率 np.ndarray(?, 10) 
        print('\n### Predicted Labels: %s ###' % (self.sess.run(tf.argmax(pre_x,1)))) # 印出模型預測的label
        print('\n### Real label: %s ###' % (self.sess.run(tf.argmax(test_y,1))))# 印出實際輸入的label


        correct_predcition = tf.equal(tf.argmax(pre_x,1), tf.argmax(test_y,1))# type(tf.argmax(pre_x,1)): ndarray
        # print('correct_prediction:', self.sess.run(correct_predcition))# 印出模型預測與實際輸入的比對結果
        

        accuracy = tf.reduce_mean(tf.cast(correct_predcition, tf.float32))
        print('\n### accuracy rate: %s ###\n' % (self.sess.run(accuracy)))# 印出正確率



    def save(self, path='./For_transfer_learning/model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)


def train():
    extracting_start = time.time()
    
    imgs, labels = load_data()
    vgg = Vgg16(vgg16_npy_path='./For_transfer_learning/vgg16.npy')
    print('\n### Net built ###')

    extracting_end = time.time()
    print('\n### The spending time of extracting is %s ###\n' % (extracting_end - extracting_start))
    
    train_start = time.time()
    for k in range(501):
        
        random_idx= np.random.randint(0, (100 * category), 10)
        
        # train_start_one = time.time()
        count = 0
        for i in random_idx:
            divisor = i // 100
            remaindor = i % 100
            content = train_file + str(divisor)

            if count == 0:
                xs = imgs[content][remaindor]
                ys = labels[content][remaindor]
                count = 1
            else:
                xs = np.concatenate((xs, imgs[content][remaindor]), axis = 0)# np.ndarray(10,224,224,3)]
                ys = np.concatenate((ys, labels[content][remaindor]), axis = 0)# np.ndarray(10,10)
            
        train_loss = vgg.train(xs, ys)

        # train_end_one = time.time()
        # print('\n### The spending time of training one time is %s ###' % (train_end_one - train_start_one))
        
        if k % 100 == 0:
            print('### steps: %s, loss: %s ###'% (k, train_loss))

    train_end = time.time()

    print('\n### The spending time of training is %s ###\n' % (train_end - train_start))
    vgg.save('./For_transfer_learning/model/transfer_learn') 


def eval():
    vgg = Vgg16(vgg16_npy_path='./For_transfer_learning/vgg16.npy',
                restore_from='./For_transfer_learning/model/transfer_learn')
    vgg.predict(
        test_path, labels = [test_label,category]) # 種類
        # labels[x,y], x: 第0~9號, y: 總共幾種


if __name__ == '__main__':
    # get_train_data(train_cap_path)
    train()
    # get_test_data(test_cap_path,waitkey = 50, number = 33)
    # eval()

# 優化程式內容，讓整體更簡潔
# 單次測試時間為0.18秒，總共訓練時間為185秒，正確率可以達到96%
# Debug: local variable 'output' referenced before assignment 訓練資料裡面沒東西