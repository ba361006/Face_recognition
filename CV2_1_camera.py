#使用USB錄影機
import cv2
import numpy as np


cap = cv2.VideoCapture(0)#0為預設 總共可以用0~4號攝影機

# print(cap.isOpened())如果打不開可以用這段程式檢查攝影機有沒有開；True or Fulse
#cap.open()若沒有開起就跟用這行開啟

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)#取得照片的寬度
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)#取得照片的長度
print("Image Size: %d x %d" % (width,height))#印出照片長寬


while(True):#一個無限迴圈，只是為了讓輸入的圖片串流成影片
    ret, frame = cap.read() #讀取一張畫面；ret為每次攝影機傳回圖片成功與否，frame為傳回的圖片

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#把照片轉灰階
    

    cv2.imshow('output',frame)#秀出名為'output'的視窗，視窗內的串流影片為frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #按q關掉視窗，一定要小寫，因為是ASCII
        break


cap.release() #釋放攝影機
cv2.destroyAllWindows() #關閉所有OpenCV的視窗



