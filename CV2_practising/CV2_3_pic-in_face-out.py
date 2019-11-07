#輸入一張圖片，對其做臉部辨識後儲存

import cv2


img = cv2.imread('D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/For_transfer_learning/data/no_5/0001_01.jpg')
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,640) #設定視窗寬度
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360) #設定視窗長度

#cv2.CascadeClassifier從OpenCV預設的演算法中使用正面臉部辨識，其演算法是使用haar(哈爾分類法)
faceCascade = cv2.CascadeClassifier('D:/Tools/Anaconda/envs/tensorflow/Library/etc/haarcascades/haarcascade_frontalface_alt2.xml')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #把圖片轉灰階

#detectMultiScale臉部辨識的傳回值分別為(x座標,y座標,寬度,長度)；(要辨識的圖片, 放大的倍率, 找的格數, 最小的臉部大小)建議上網查
faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30))
for (x,y,w,h) in faces: #把傳回值分別存到x,y,w,h
    cv2.rectangle( #在圖片上面畫矩形，；(要畫的圖片, 第一個角落, 第二個角落, BGR, 線條寬度)
                   #這裡要注意角落座標可為圖片左上右下或右上左下，都可以劃出同一個矩形
        img,
        (x-10,y-10),
        (x+10+w,y+10+h),
        (0,255,0),
        2
         )
output = img[y:y+h,x:x+w] #把img於[col,row]存到output內



# cv2.imshow('img',img)
cv2.imwrite('Captures/picture.jpg', output)

# cv2.waitKey(0)
# cv2.destroyallWindows()
    

