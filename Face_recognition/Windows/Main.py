import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import os
import numpy as np
import skimage.io
import skimage.transform
import collections 
import cv2
import time 
import shutil
import threading


force_in = 'D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/Reinforcement/'
train_data_path = 'D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/Train_data/'
train_batch = 10
keep_rate = 0.5
train_learning_rate = 5e-4
sample_steps = 100
train_steps = 501 
augment = 101

category = len(os.listdir(train_data_path))
Test_path = 'D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/Test_data/test_data/'
shared_path = 'D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/shared/'
def load_data():

    dict_imgs = build_dict(train_data_path)
    dict_labels = build_dict(train_data_path)
    count = 0

    for  k in dict_imgs.keys(): 
        labels = np.zeros([1,len(dict_imgs)])
        labels[0][count] = 1 
        dir = train_data_path + k
        print('\n### Loading %s ###\n' % (k))
    

        for file in os.listdir(dir):
            if not file.lower().endswith('.jpg'):
                continue
            try:
                resized_img = load_img(os.path.join(dir, file))
            except OSError:
                continue
            
            dict_imgs[k].append(resized_img)
            dict_labels[k].append(labels) 
            if len(dict_imgs[k]) == sample_steps:
                break
        count += 1
    return dict_imgs, dict_labels



def build_dict(path): 
    dict_name = collections.OrderedDict()
    for file in os.listdir(path):
        dict_name[file] = []
    return dict_name



def load_img(path):
    img = skimage.io.imread(path)
    img = img / 255.0
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224, 3))[None, :, :, :]   
    return resized_img

def delete(path):
    folder_path = os.path.exists(path)
    if folder_path:
        shutil.rmtree(path)
    time.sleep(0.1)
    os.makedirs(path)

def file2img(input):
    i = 0 
    for file in os.listdir(input):
        img_path = os.path.join(input, file)
        if i == 0:
            output = load_img(img_path)
            i = 1
        else:
            output = np.concatenate((output, load_img(img_path)), axis = 0)
    return output


def ok(user_name):
    global category
    Train_path = 'D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/Train_data/' + user_name + '/'
    delete(Train_path)
    category = len(os.listdir(train_data_path))
    

def capture_training_data(user_name, output_path = force_in, waitkey = 25, number = 1):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
    
    count = 0
    while(True):
        faceCascade = cv2.CascadeClassifier('D:/Tools/Anaconda/envs/tensorflow/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2, minNeighbors = 4, minSize = (30,30))
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(
                frame,
                (x - 20,y - 20),
                (x + 20 + w,y + 20 + h),
                (0,255,0),
            )
        cv2.imshow('frame',frame)

        count += 1
        try:
            if cv2.waitKey(10) & count == 20: 
                delete(output_path)
                Train_path = train_data_path + user_name + '/'
                for num_0 in range(number):

                    ''' Uncomment here if you want to take more than one picture.
                    ret, frame = cap.read()
                    for (x,y,w,h) in faces:
                        frame = cv2.rectangle(
                            frame,
                            (x - 20,y - 20),
                            (x + 20 + w,y + 20 + h),
                            (0,255,0),
                        )
                    cv2.imshow('frame',frame)
                    cv2.waitKey(waitkey)
                    '''

                    img = frame[y:y+h,x:x+w]
                    cv2.imwrite(output_path + 'face_' + str(num_0) + '.jpg', img) 
                    # print('\n### Taking %s picture ###' % (num_0))

                    if num_0 == (number - 1):
                        print('\n### Finish Capturing ###\n')
                        print('\n ### Train_path %s ###' %(Train_path))
                        reinforcement(force_in, Train_path, user_name, augment)
                cap.release()
                cv2.destroyAllWindows()
                break
        except UnboundLocalError:
            print("Can't find your face, please try again !")
            cap.release()
            cv2.destroyAllWindows()
            break



def capture_testing_data(output_path = Test_path, waitkey = 50, number = 10):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)
    count = 0
    while(True):
        faceCascade = cv2.CascadeClassifier('D:/Tools/Anaconda/envs/tensorflow/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2, minNeighbors = 4, minSize = (30,30))
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(
                frame,
                
                (x - 20,y - 20),
                (x + 20 + w,y + 20 + h),
                (0,255,0),
                
            )
        cv2.imshow('frame',frame)

        count += 1
        try:
            if cv2.waitKey(10) & count == 20: 
                delete(output_path)
                for num_1 in range(number):

                    ret, frame = cap.read()
                    for (x,y,w,h) in faces:
                        frame = cv2.rectangle(
                            frame,
                            (x - 20,y - 20),
                            (x + 20 + w,y + 20 + h),
                            (0,255,0),
                        )
                    cv2.imshow('frame',frame)
                    cv2.waitKey(waitkey)
                    
                    img = frame[y:y+h,x:x+w]
                    cv2.imwrite(output_path + 'face_' + str(num_1) + '.jpg', img) 
                    # print('\n### Taking %s picture ###' %(num_1))
                    if num_1 == (number - 1):
                        print('\n### Finish Capturing ###\n')
                cap.release()
                cv2.destroyAllWindows()
                break
        except UnboundLocalError:
            print("Can't find your face, please try again !")
            cap.release()
            cv2.destroyAllWindows()
            break

        


def reinforcement(input_path , output_path, output_name, number):
    datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,)
    # print('\n### Reinforcing pictures to: %s ###\n' %(output_path))
    delete(output_path)
    k = 0
    for file in os.listdir(input_path):
        k += 1
        path = input_path + file
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((1,)+img.shape)
        i = 0
        for _ in datagen.flow(img, batch_size = 2, save_to_dir = output_path, save_prefix = output_name, save_format = 'jpg'):
            i += 1
            if i == number:
                break
    
    print('\n### %s pictures have been reinforced into %s pictures ####\n'% (k, (k * number)) )
    

def show_result(name, accuracy):
    delete(shared_path)
    time.sleep(0.1)
    background = np.zeros((400,400,3), np.uint8)
    text = ''
    if name == 'Others' or accuracy < 0.9:
        background[:,:,:] = (0,0,130)
        text = 'Error, please try again'
        cv2.putText(background, text, (60,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
    else:
        background[:,:,:] = (0,130,0)
        text = 'Hello, ' + str(name) + '!'
        cv2.putText(background, text, (110,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)
    
    
    cv2.imwrite(shared_path + '/' + str(text) +'.jpg',background)
    # cv2.imshow('Hello!', background)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


def entering_list():
    list_path = 'D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/List/'
    note = os.listdir('D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/shared')
    entrance = note[0].split(', ')[0]

    localtime = time.asctime( time.localtime(time.time()) )
    name_list = open(list_path + 'name_list.txt', 'a')
    
    if entrance == 'Hello':
        visitor = note[0].split(', ')[1].split('!')[0]
        name_list.write('\n' + localtime + ' ' + visitor + ' enters\n' )
    else:
        name_list.write('\n' + localtime + ' Error, someone is trying to enter your house!\n' )


class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]
    
    def __init__(self, vgg16_npy_path=None, restore_from=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('\n### Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr ###\n')

        self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.tfy = tf.placeholder(tf.float32, [None, category])
        self.keep_prob = tf.placeholder(tf.float32)
        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        self.conv_count = 0
        
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

        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01), name='fc6')
        self.drop_1 = tf.nn.dropout(self.fc6, self.keep_prob, name = 'drop_1')

        self.out = tf.layers.dense(self.drop_1, category ,kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01), name='out') #  種類
        self.drop_2 = tf.nn.dropout(self.out, self.keep_prob, name = 'drop_2')

        self.test_out = tf.nn.softmax(self.drop_2)

        self.sess = tf.Session()
        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.out, labels = self.tfy ))
            self.train_op = tf.train.AdamOptimizer(train_learning_rate).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)



    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict = {self.tfx: x, self.tfy: y, self.keep_prob: keep_rate})
        return loss

    def predict(self, paths):#paths, labels = list
        test_x = file2img(paths)
        # print('\n### Shape of test_data: %s ###' % (test_x.shape,))
        print('\n### Predicting ###\n')


        pre_x = self.sess.run(self.test_out, feed_dict = {self.tfx: test_x, self.keep_prob: 1})
        pre_label = self.sess.run(tf.argmax(pre_x,1))
        print('\n### Tenants: %s ###' % (list(enumerate(os.listdir(train_data_path)))))
        print('\n### Predicted label: %s ###' % (pre_label))


        most_common_num = collections.Counter(pre_label).most_common(1)[0][1]
        most_common_label = collections.Counter(pre_label).most_common(1)[0][0]
        accuracy = most_common_num / len(pre_label)
        print('\n### Accuracy rate: %s ###\n' % (accuracy))

        predicted_name = os.listdir(train_data_path)[most_common_label]
        print('\n### show result ###\n')
        show_result(predicted_name, accuracy)


    def compute_accuracy(self, xs, ys):
        pre_x = self.sess.run(self.test_out, feed_dict = {self.tfx: xs, self.keep_prob: 1})
        pre_label = self.sess.run(tf.argmax(pre_x,1))
        print('\n#### pre_label: %s, true_label: %s' %(pre_label, self.sess.run(tf.argmax(ys,1))))

        correct_prediction = tf.equal(pre_label, self.sess.run(tf.argmax(ys,1)))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        print('\n### Accuracy rate: %s ###\n' % (self.sess.run(accuracy, feed_dict = {self.tfx: xs, self.keep_prob: 1})))


    def save(self, path='D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/Transfer_learning/model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)



def train():
    global category
    category = len(os.listdir(train_data_path)) 
    tf.reset_default_graph()
    imgs, labels = load_data()
    vgg = Vgg16(vgg16_npy_path='D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/Transfer_learning/vgg16.npy')
    print('\n### Net built ###')

    folders = os.listdir(train_data_path)
    train_start = time.time()
    for k in range(train_steps):
        random_idx= np.random.randint(0, (sample_steps * category), train_batch)
        count = 0
        for i in random_idx:
            divisor = i // sample_steps - 1
            remaindor = i % sample_steps
            content = str(folders[divisor])
            if count == 0:
                xs = imgs[content][remaindor]
                ys = labels[content][remaindor]
                count = 1
            else:
                xs = np.concatenate((xs, imgs[content][remaindor]), axis = 0)# np.ndarray(10,224,224,3)]
                ys = np.concatenate((ys, labels[content][remaindor]), axis = 0)# np.ndarray(10,10)
            
        train_loss = vgg.train(xs, ys)
        # loss_txt = open('D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/List/loss.txt', 'a') # Record loss
        # loss_txt.write(str(train_loss) + '\n')
        if k % sample_steps == 0:          
            print('### steps: %s, loss: %s ###'% (k, train_loss))  
            # print(vgg.compute_accuracy(xs,ys))
    train_end = time.time()
    print('\n### Finish training ###\n')
    print('### The spending time of training is %s ###\n' % (train_end - train_start))
    vgg.save('D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/Transfer_learning/model/transfer_learn') 



def eval():
    tf.reset_default_graph()
    vgg = Vgg16(vgg16_npy_path='D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/Transfer_learning/vgg16.npy',
                restore_from='D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/Transfer_learning/model/transfer_learn')
    vgg.predict(Test_path)
    entering_list()
