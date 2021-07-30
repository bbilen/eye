# USAGE
# python eyeblinking.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat

import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import I2C_LCD_driver
from gpiozero import LED
from imutils.video import VideoStream
from imutils import face_utils

mylcd = I2C_LCD_driver.lcd()
mylcd.lcd_clear()
led1=LED(21)
led2=LED(16)
mylcd.lcd_display_string("Welcome",1)

def euclidean_dist(ptA, ptB):

    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)

    return ear

def kelime(left_blink,right_blink):
    mylcd.lcd_clear()
    if left_blink==0:
        thistuple = ("Merhaba", "Evet", "Su","Naber","Televizyon")
        mylcd.lcd_display_string(thistuple[right_blink],1)

    if left_blink==1:
        thistuple = ("Hayir", "Yemek", "iyiyim","Tuvalet","Olur")
        mylcd.lcd_display_string(thistuple[right_blink],1)
    
    if left_blink==2:
        thistuple = ("Tamam", "Uyku", "Sicak","Ben")
        mylcd.lcd_display_string(thistuple[right_blink],1)
        
    if left_blink==3:
        thistuple = ("Nasilsin", "Soguk","Sen")
        mylcd.lcd_display_string(thistuple[right_blink],1)
 
    if left_blink==4:
        thistuple = ("Saat","Para")
        mylcd.lcd_display_string(thistuple[right_blink],1)
        
        
    time.sleep(3)
    return  

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,help = "path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True,help="path to facial landmark predictor")
args = vars(ap.parse_args())
# check to see if we are using GPIO/TrafficHat as an alarm


EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES =3

COUNTER_RIGHT = 0
COUNTER_LEFT = 0
left_blink=0
right_blink=0

mylcd.lcd_display_string("running...",1)

print("[INFO] loading facial landmark predictor...")
detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")
mylcd.lcd_display_string("starting in 2secs",1)

vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
mylcd.lcd_clear()
#vs.framerate=16

now = time.time()
future = now + 10

while True:
    
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    cv2.putText(frame, "Wink Left : {}".format(left_blink), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Wink Right: {}".format(right_blink), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR) / 2.0
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "E.A.R. Left : {:.2f}".format(leftEAR), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "E.A.R. Right: {:.2f}".format(rightEAR), (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if leftEAR <=EYE_AR_THRESH and leftEAR<rightEAR:
            COUNTER_LEFT += 1
        else:    
            if COUNTER_LEFT >= EYE_AR_CONSEC_FRAMES:
                left_blink+=1
                led2.on()
            COUNTER_LEFT=0
                
        if rightEAR <= EYE_AR_THRESH and rightEAR<leftEAR:
            COUNTER_RIGHT +=1
        else:
            if COUNTER_RIGHT >= EYE_AR_CONSEC_FRAMES:
                right_blink+=1
                led1.on()
            COUNTER_RIGHT=0
        mylcd.lcd_display_string("Left Blink:%d" %(left_blink),1)
        mylcd.lcd_display_string("Right Blink:%d" %(right_blink),2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
        
    if key == ord("q"):
         break
    led1.off()
    led2.off()

    if time.time() > future:
        COUNTER_LEFT=0
        COUNTER_RIGHT=0
        print("10sn")
        kelime(left_blink, right_blink)
        right_blink=0
        left_blink=0
        now = time.time()
        future = now + 10 
       

cv2.destroyAllWindows()
vs.stop()
