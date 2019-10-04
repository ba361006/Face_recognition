from __future__ import absolute_import, division, print_function

#TensorFlow and tf.keras
#keras是一個在tensorflow裡面的API
import tensorflow as tf
from tensorflow import keras

#Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
#print1



fashion_mnist = keras.datasets.fashion_mnist
#引入一個在Keras裡面的Datasets 裡面有七萬筆資料分別有十個種類 (28 by 28 pixels
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# XXX.load_data() 讀資料



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#種類名稱把它儲存在一個陣列裡面 且若 print(class_names[0])會印出來 T-shirt/top


print(train_images.shape)
#可以看出來image 位元數,pixels 
#print2

print(len(train_labels))
#len是指train_labels裡面有多少筆資料 
#print3

print(train_labels)
#印出裡面的內容
#print4

# test = plt.imread("test.jpg")


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#=====================                  範例                      ==================
# import matplotlib.pyplot as plt
# import numpy as np

# x = np.linspace(-3, 3, 50)
# y1 = 2*x + 1
# y2 = x**2

# plt.figure()                                    figure代表一張圖
# plt.plot(x, y1)                                 plt.plot代表這張圖的內容是甚麼

# plt.figure(num=3, figsize=(8, 5),)              這裡的plt.figure代表第二章圖，名字是三，大小是8,5
# plt.plot(x, y2)                                 一條線
# # plot the second curve in this figure with certain parameters
# plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')                   
# 第二條線的顏色、線徑、虛線
# plt.show()                                      把圖SHOW出來
#======================================================================================


train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
        plt.subplot(5,5,i+1)
        #把多張圖合併，x軸有五張，y軸有五張，圖放在第幾個位置順序是由上至下由左至右
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        #cmap=plt.mp.binary是指用灰階的Colorbar作圖
        plt.xlabel(class_names[train_labels[i]])
plt.show()
        



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,train_labels, epochs=5)



test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy', test_acc)


predictions = model.predict(test_images)
predictions[0]


np.argmax(predictions[0])


test_labels[0]

def plot_image(i, predictions_array, true_label, img):
        predictions_array, true_label, img = predictions_array[i],true_label[i], img[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        
        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
                color = 'blue'
        else:
                color = 'red'
        
        plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                             100*np.max(predictions_array),
                                             class_names[true_label]),
                                             color = color)


def plot_value_array(i, predictions_array, true_label):
        predictions_array, true_label = predictions_array[i], true_label[i]
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array, color="#777777")
        plt.ylim([0,1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')

i=0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()



# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)
plt.show()


# Grab an image from the test dataset
img = test_images[0]
print(img.shape)


# Add the image to a batch whhere it's the only member.
img = (np.expand_dim(img,0))
print(img.shape)


predictions_single = model.predict(img)
print(predictions_single)


plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)


np.argmax(predictions_single[0])

