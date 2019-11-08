#按照拍照順序命名儲存圖片
import cv2
num = 1



cap = cv2.VideoCapture(0)

width = cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
height = cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)


while(True):
    faceCascade = cv2.CascadeClassifier('D:/Tools/Anaconda/envs/tensorflow/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2, minNeighbors = 7, minSize = (30,30))
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(
            frame,
            
            (x - 10,y - 10),
            (x + 10 + w,y + 10 + h),
            (0,255,0),
            2
            
        )
    cv2.imshow('frame',frame)


    if cv2.waitKey(10) & 0xFF == ord('c'): 
        img = frame[y:y+h,x:x+w]
        cv2.imwrite("Captures/face" + str(num) + '.jpg', img) #按照儲存順序命名檔案
        num += 1 #這裡是利用字串的方式對'face'加上'數字'以及'jpg'

    elif cv2.waitKey(10) & 0xFF == ord('q'):
        break

    
cap.release()
cv2.destroyAllWindows()