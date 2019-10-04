#把錄影機的內容輸出成影片
import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,360)

fourcc = cv2.VideoWriter_fourcc(*'XVID')#設定影像格式

#儲存影片到" "內的位置；cv2.VideoWriter(檔案位置/檔名.__ ,影像格式,FPS,影像尺寸)
out = cv2.VideoWriter("Videos/Camera2_output_fps60.avi", fourcc, 20.0, (640,360))

while(True):

    ret, frame = cap.read()
    
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    out.write(frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()


    


