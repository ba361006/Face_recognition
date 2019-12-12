#輸入一張圖片，對其做臉部辨識後儲存

import cv2
import numpy as np 
import os 

def Capture(input_path, output_path):
    count = 0
    for path in os.listdir(input_path):
        img = cv2.imread(input_path + '/' + path)
        faceCascade = cv2.CascadeClassifier('D:/Tools/Anaconda/envs/tensorflow/Library/etc/haarcascades/haarcascade_frontalface_alt2.xml')
        faces = faceCascade.detectMultiScale(img, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30))
        for (x,y,w,h) in faces: #把傳回值分別存到x,y,w,h
            cv2.rectangle( #在圖片上面畫矩形，；(要畫的圖片, 第一個角落, 第二個角落, BGR, 線條寬度)
                        #這裡要注意角落座標可為圖片左上右下或右上左下，都可以劃出同一個矩形
                img,
                (x - 20, y - 20),
                (x + 20 + w, y + 20 + h),
                (0,255,0),
                2
                )
        output = img[y:y+h,x:x+w]
        cv2.imwrite(output_path + path, output)
        count += 1
        print('Capturing picture ', count)

    print('Finish Processing')

Capture(input_path = 'D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/face/', output_path = 'D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/Project/hello/') 