import RPi.GPIO as GPIO
import Project_communicate as comm
import time
import os
import shutil


CONTROL_PIN = 17 
BUTTON_PIN = 18
MS_SWITCH = 20

flag = 0
flag2 = 1

PWM_FREQ = 50




def angle_to_duty_cycle(angle=0):
	duty_cycle = (0.05 * PWM_FREQ) + (0.19 * PWM_FREQ * angle / 180)
	return duty_cycle

def call_sg90_on():
    
    dc = angle_to_duty_cycle(120)
    pwm.ChangeDutyCycle(dc)
    time.sleep(0.5)
  
def call_sg90_off():
    
    dc = angle_to_duty_cycle(50) 
    pwm.ChangeDutyCycle(dc)
    time.sleep(0.5)
    
'''
def Button_1(channel): 
    global flag2, flag
    print("Button1")
    time.sleep(0.1)
    flag2 =1
    
    # key = comm.start()
    # print('\n### key: %s ###\n' %(key) )
    # time.sleep(0.1)
    # if key == 'Hello':
    call_sg90_off()
        # time.sleep(2)
    flag = 1
'''
        





GPIO.setmode(GPIO.BCM)

GPIO.setwarnings(False)

GPIO.setup(BUTTON_PIN, GPIO.IN,pull_up_down=GPIO.PUD_UP )
GPIO.setup(MS_SWITCH, GPIO.IN,pull_up_down=GPIO.PUD_UP )

GPIO.setup(CONTROL_PIN, GPIO.OUT)

pwm = GPIO.PWM(CONTROL_PIN, PWM_FREQ)
pwm.start(0)


#GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=Button_1, bouncetime = 250)

try:
    while True:
        
        if ((GPIO.input(BUTTON_PIN) == GPIO.LOW) & flag2):
            flag2 = 0
            print("button 1")
            time.sleep(0.05)
            key = comm.start()
            print('\n### key: %s ###\n' %(key) )
            time.sleep(0.1)
            if key == 'Hello':
                call_sg90_off()
                time.sleep(2)
                flag = 1
            flag2 = 1
            

            
        
        
        if (GPIO.input(MS_SWITCH) == GPIO.HIGH) :
            if flag:
                print("switch")
                time.sleep(2)
                call_sg90_on()
                flag = 0

        
        time.sleep(0.3)
except KeyboardInterrupt:
    print("shunt down")
finally:
    GPIO.cleanup()

