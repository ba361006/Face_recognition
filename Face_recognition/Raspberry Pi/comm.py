import cv2
import os
import time
import paramiko 
import shutil
from ftplib import FTP


# FTP parameters
FTP_local_ip = ''  
FTP_local_port = 21
FTP_local_user = ''
FTP_local_password = ''


FTP_pi_ip = ''   
FTP_pi_port = 21
FTP_pi_user = 'pi'
FTP_pi_password = ''

FTP_pc_path = '/Test_data/test_data/'
FTP_pc_shared = '/shared/'

FTP_pi_path = '/home/pi/opencv-3.4.3/shared/face/'
FTP_pi_result = '/home/pi/opencv-3.4.3/shared/result/'

def delete(path):
    folder_path = os.path.exists(path)
    if folder_path:
        shutil.rmtree(path)
    os.makedirs(path)

def capture_testing_data(output_path, waitkey = 1, number = 10):
    count = 0
    start = time.time()
    cap = cv2.VideoCapture(0)
    while(True):
        count += 1
        faceCascade = cv2.CascadeClassifier('/home/pi/opencv-3.4.3/data/haarcascades_cuda/haarcascade_frontalface_default.xml')
        _, frame = cap.read()
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

        try :  
            if cv2.waitKey(1) & count == 10:
                delete(FTP_pi_path)
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
                    # print('\n### Taking %s picture ###' % (num_1))
                    if num_1 == (number - 1):
                        # print('\n### Finish Capturing ###\n')
                        end = time.time()
                        print('\n### Capture spending time: %s ###\n' % (end - start))
                cap.release()
                cv2.destroyAllWindows()
                break
        except UnboundLocalError:
            print("Can't find your face, please try again !")
            count = 0
            pass




def upload(server_path, remote_path):
    start = time.time()

    ftp=FTP() 
    ftp.set_debuglevel(2) 
    ftp.connect(FTP_local_ip, FTP_local_port)
    ftp.login(FTP_local_user, FTP_local_password) 
    print('\n### Uploading ###\n')


    for data in os.listdir(server_path):
        file_name = open(server_path + data, 'rb')
        ftp.storbinary('STOR ' + remote_path + data, file_name)
    file_name.close()
    ftp.quit()

    end = time.time()
    print('\n### Upload spending time: %s ###\n' % (end - start))



def SSH():
    start = time.time()
    time.sleep(0.1)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())', 
    ssh.connect(hostname = '' ,
    port = 22,
    username = '',
    password = '')
    
    # Run
    print('\n### SSH Processing ###')
    _, stdout, stderr = ssh.exec_command('conda run -n tensorflow python D:/Tools/Tensorflow_projects/Test1_MNIST_VSCode/eval.py')
    activate_out = stdout.readlines()
    # activate_err = stderr.read()
    print('\n### activate_out: ###\n', activate_out) 
    # print('\n### activate_err: ###\n', activate_err)
    time.sleep(0.1)
    _,_,_ = ssh.exec_command('exit')
    ssh.close()
    
    end = time.time()
    print('\n### SSH spending time: %s ###\n' % (end - start))



def download(server_path, remote_path):
    start = time.time()

    delete(server_path)
    ftp=FTP() 
    ftp.set_debuglevel(2) 
    ftp.connect(FTP_local_ip, FTP_local_port)
    ftp.login(FTP_local_user, FTP_local_password) 
    print('\n### Downloading ###\n')

    ftp.cwd('/shared')# enter the specific path
    filenames = []
    ftp.retrlines('NLST', filenames.append)


    for number in range(len(filenames)):
        server_file = open(server_path + filenames[number], 'wb')
        print('\n### server_path + filenames[number]: %s, %s ###\n' % (remote_path + filenames[number], type(server_path + filenames[number])) )
        ftp.retrbinary('RETR ' + remote_path + filenames[number], server_file.write, 1024)
    server_file.close()
    ftp.quit()

    end = time.time()
    print('\n### Download spending time: %s ###\n' % (end - start))



def img_show():
    file_name = os.listdir('/home/pi/opencv-3.4.3/shared/result/')
    data = '/home/pi/opencv-3.4.3/shared/result/' + file_name[0]
    key = file_name[0].split(',')[0]
    img= cv2.imread(data, cv2.IMREAD_COLOR)
    cv2.imshow('Hello',img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    return key
    



def start():
    all_start = time.time()

    capture_testing_data(output_path = FTP_pi_path)
    upload(server_path = FTP_pi_path, remote_path = FTP_pc_path)
    SSH()
    download(server_path = FTP_pi_result, remote_path = FTP_pc_shared)
    key = img_show()

    all_end = time.time()
    print('\n### Total spending time: %s ###\n' % (all_end - all_start))
    return key








