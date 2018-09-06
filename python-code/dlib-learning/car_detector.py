import dlib
import cv2
# import cvutils 
import sys
import os
import time
from skimage import io
#load car detector
detector = dlib.fhog_object_detector("../resources/models/dlib/car_detector.svm")
win = dlib.image_window()
#give path of video on which you want it to run and process frame by frame
#if you want to use my video then download that from here 
#https://drive.google.com/file/d/19iE0RuCi9uVm_xLjOuG7fYkRfLktYDis/view?usp=sharing
cap = cv2.VideoCapture('../resources/videos/car_video2.avi')
while(True):
#     # Capture frame-by-frame
     ret, frame = cap.read()
     if not ret:
         continue
     #process in BGR2RGB for more detections
     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

     dets = detector(frame)
    

     for d in dets:
         cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
        
#     # Display the resulting frame
     cv2.imshow("frame",frame)
     if cv2.waitKey(1) & 0xFF == ord('q'):
         break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()