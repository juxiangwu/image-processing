from imutils import face_utils #for resizing
import numpy as np
import argparse
import imutils
import dlib
import cv2
import time
from scipy.spatial import distance as dist #euclidian distance
import csv
from pathlib import Path
from picamera.array import PiRGBArray
from picamera import PiCamera
import picamera
import picamera.array
import os

csv_columns = ['name', 'face_data']
csv_file = 'all_face_data.csv'


camera_id = 0
EAR_AR_THRESH = 0.3
CONSEC_FRAMES = 1
TOTAL = 0
COUNT = 0
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print('App is preparing, please wait.')
detector = dlib.get_frontal_face_detector() # detect the faces in the image. How many faces are there
predictor = dlib.shape_predictor('../resources/models/dlib/shape_predictor_68_face_landmarks.dat') # predict the face landmarks such as mouth or eyes
facerec = dlib.face_recognition_model_v1('../resources/models/dlib_face_recognition_resnet_model_v1.dat') #pretrained model. 
#we send the data to this function and it returns a 128D vector that described the faces.



def eye_aspect_ratio(eye):
    #https://www.pyimagesearch.com/wp-content/uploads/2017/04/blink_detection_plot.jpg
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    
    c = dist.euclidean(eye[0], eye[3])
    
    ear = (a+b) / (2.0 * c)
    
    return ear


def write_dict_to_csv(csv_file, csv_columns, dict_data):
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            for key, value in dict_data.items():
                writer.writerow({'name': key, 'face_data': value})
    except IOError:
        print("I/O error", csv_file)
    return


def append_to_csv(csvfile, data):
    with open(csvfile, 'a') as f:
        writer = csv.writer(f)
        for key, value in data.items():
            writer.writerow([key,value])
    return


def cvt_to_array(data, split_with=''):
    if split_with == '':
        return np.array(list(map(float, data)))
    else:
        return np.array(list(map(float, data.split(split_with))))


def menu():
    print('Welcome to face recognition system. What do you want to do?\n')
    action = 0
    while action != 3:
        print('1) Run the app\n')
        print('2) Save new face to database\n')
        print('3) Exit\n')
        action = int(input('>>'))
        if action == 1:
            
            while True:
                run_app()
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        elif action == 2:
            save_face()
            print('App must be restarted after the new face saved...')
            #quit()
    


def run_app():
    global COUNT, TOTAL
    found_face = 0
    with picamera.PiCamera() as camera:
        camera.start_preview()
        with picamera.array.PiRGBArray(camera) as stream:
            camera.capture(stream, format='bgr')
        	# At this point the image is available as stream.array
            image = stream.array
            image = imutils.resize(image, width=300)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if found_face % 5 == 0:
            	rects = detector(gray, 0)
            	print('Searching for the faces...')
            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the landmark (x, y)-coordinates to a NumPy array
                shape = predictor(gray, rect)
                print('Predicting the faces...')
                trying = np.array(facerec.compute_face_descriptor(image, shape))
                with open("./" + csv_file, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row == [] or row[1] == "face_data":
                            continue
                        else:
                           
                            row[1] = cvt_to_array(row[1], '\n')
                            trying = cvt_to_array(trying)
                            distance_faces = dist.euclidean(row[1], trying)
                            if distance_faces < 0.55:
                                content = row[0]
                                content = 'Access Granted! Welcome ' + str(content)
                                
                                break
                            else:
                                content = "Unknown person detected!"
                                
                print(content)
            
                time.sleep(3)
                shape = face_utils.shape_to_np(shape)
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
           
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
    
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)
 

def save_face():
    

    # capture the person and save as the 128D vector
    # this part captures only once, if you want to save another face, just call this function again.


    camera = PiCamera()
    rawCapture = PiRGBArray(camera)	

    face_data = []
    labels = []
    data = {}


    face_number = 0
    while face_number == 0:
        print('Please show your whole face to camera. When the face is detected, you will be asked for the name.')
        time.sleep(0.5)
        camera.capture(rawCapture, format="bgr")
        image = rawCapture.array
        image = imutils.resize(image, width=500) #resizing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #it should convert to gray in onder to improve resultt.
        rects = detector(gray, 0) # detect how many faces in the image
        
        for (i, rect) in enumerate(rects): 
            # for every faces
            # determine the facial landmarks for the face region, then
            # convert the landmark (x, y)-coordinates to a NumPy array
            shape = predictor(gray, rect) # predict the face landmarks in image. 

            face_descriptor = facerec.compute_face_descriptor(image, shape) # send the shape data to resnet model. it returns a 128D vector
            
            while face_descriptor == -1:
                print('Face not found.')
            else:
                face_data.append(face_descriptor) # save the face data to array
                shape = face_utils.shape_to_np(shape)
                for (x, y) in shape:
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                name = str(raw_input('Who is this : '))
                while (not name.isalpha or name == ''):
                    print('Please enter a valid name.')
                    name = str(raw_input('Who is this : '))
                labels.append(name)
                data[labels[0]] = face_data[0]
                face_data=[]
                labels=[]
                my_file = Path("./" + csv_file)
                if my_file.is_file():
                    append_to_csv(csv_file, data)
                    print('File already exist, data is appended to file')
                else:
                    write_dict_to_csv(csv_file, csv_columns, data)    
                    print('File has been created and data saved to file.')
                face_number += 1
                
                camera.close()
        rawCapture.truncate(0) 



if __name__ == '__main__':
    menu()