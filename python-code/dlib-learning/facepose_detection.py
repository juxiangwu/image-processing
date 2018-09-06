# Author: Abhishek Sharma
# Program: Face Pose Estimation using Haar Cascasde, HOG and Dlib Library.

import os
import cv2
import numpy as np
import sys
import imutils
import dlib
from imutils import face_utils

class PoseDetection:

    def __init__(self,option_type,path):

        self.face_cascade = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier("cascade/haarcascade_eye.xml")
        self.smile_cascade = cv2.CascadeClassifier("cascade/haarcascade_smile.xml")
        self.shape_predictor = "cascade/shape_predictor_68_face_landmarks.dat"
        self.facedetect = False
        self.functioncall = option_type
        self.sourcepath = path
        self.image_path = None
        self.video_path = None
        self.webcam_path = None
        self.main_function()

    def haar_facedetection(self,img):
        faces = self.face_cascade.detectMultiScale(img,1.3,5)
        print(faces)
        return faces

    def haar_eyedetection(self,img):
        eyes = self.eye_cascade.detectMultiScale(img,1.3,5)
        return eyes
        
    def haar_smilecascade(self,img):
        smile = self.smile_cascade.detectMultiScale(img,1.3,5)
        return smile
    
    def dlib_function(self,image):
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.shape_predictor)
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(image, 1)
        
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        return image


    def webcam(self):
        cap = cv2.VideoCapture(int(self.webcam_path))
        tracker = cv2.Tracker_create("MIL")
        count = 0
        while (cap.isOpened()):
            ret, img = cap.read()
            if not ret:
                print("Cannot Read Video File")
                break
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(21,21),0)
            fullbody = self.HogDescriptor(gray)
            for (x,y,w,h) in fullbody:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            faces = self.haar_facedetection(gray)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = self.haar_eyedetection(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)
                smile = self.haar_smilecascade(roi_gray)
                for (sx,sy,sw,sh) in smile:
                    cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh),(0,255,0),2)

            img = self.dlib_function(img)
            cv2.imshow('img',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def video(self):
        cap = cv2.VideoCapture(str(self.webcam_path))
        tracker = cv2.Tracker_create("MIL")
        count = 0
        while (cap.isOpened()):
            ret, img = cap.read()
            if not ret:
                print("Cannot Read Video File")
                break
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray,(21,21),0)
            faces = self.haar_facedetection(gray)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = self.haar_eyedetection(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2)
                smile = self.haar_smilecascade(roi_gray)
                for (sx,sy,sw,sh) in smile:
                    cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh),(0,255,0),2)
            img = self.dlib_function(img)
            cv2.imshow('img',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    
    def image(self):

        img = cv2.imread(self.image_path)
        img = imutils.resize(img,width=min(800,img.shape[1]))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(21,21),0)
        fullbody = self.HogDescriptor(gray)
        for (x,y,w,h) in fullbody:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        faces = self.haar_facedetection(gray)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = self.haar_eyedetection(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2) 
            smile = self.haar_smilecascade(roi_gray)
            for (sx,sy,sw,sh) in smile:
                cv2.rectangle(roi_color, (sx,sy), (sx+sw,sy+sh),(0,255,0),2)
        img = self.dlib_function(img)
        cv2.imshow('img',img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    def HogDescriptor(self,image):

        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        (rects, weights) = hog.detectMultiScale(image, winStride=(5,5),padding=(16,16), scale=1.05, useMeanshiftGrouping=False)
        return rects

    def main_function(self):
        if self.functioncall == "image":
            self.image_path = self.sourcepath
            self.image()

        elif self.functioncall == "webcam":
            self.webcam_path = self.sourcepath
            self.webcam()

        elif self.functioncall == "video":
            self.video_path = self.sourcepath
            self.video()

if __name__ == '__main__':

    print("\nPose Estimation\n")
    print("\nSelect:\n 1:Image \n 2:Video \n 3:Webcam \n ")
    input_type = int(input("Choice(Number): "))
    
    if input_type == 1:
        image_path = input("Enter Absolute Image Path: ")
        PoseDetection(option_type="image",path=str(image_path))
    elif input_type == 2:
        video_path = input("Enter Video Path: ")
        PoseDetection(option_type="video",path=str(video_path))

    elif input_type == 3:
        webcam_path = input("Enter Cam Number(0:Default): ")
        PoseDetection(option_type="webcam",path=str(webcam_path))
    else:
        print("Please Select Correct Option")