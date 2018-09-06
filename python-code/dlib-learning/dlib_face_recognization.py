import cv2
import os
import face


import numpy as np

def find_faces(image, cascade_file = "haarcascade_frontalface_alt.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    i = 0
    results = []
    for (x, y, w, h) in faces:
        i += 1
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        temp = image[y:y+h, x:x+w, :]
        results.append(temp)
    return results, image

 
def camera_realtime(file, savevideo):
    camera = cv2.VideoCapture(file)
    print('Press [ESC] to quit demo')
    
    elapsed = int()
    
    cv2.namedWindow('', 0)
    _, frame = camera.read()
    height, width, _ = frame.shape
    cv2.resizeWindow('', width, height)
    
    
    if savevideo:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if file == 0:
          fps = 10
          if fps < 1:
            fps = 1
        else:
            fps = round(camera.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter('video.avi', fourcc, fps, (width, height))
    
    while camera.isOpened():
        _, frame = camera.read()
        if frame is None:
            print ('\nEnd of Video'+'\n'+'----------------------')
            break
        faces, processed = find_faces(frame)
        elapsed += 1
        if elapsed % 5 == 0:
            for each in faces:
                recognize_result = face.recognize_face(each,descriptors,names)
                if recognize_result == -1:
                    print('this person does not exit in candidate-faces')
                    cv2.putText(processed,'this person does not exit in candidate-faces',(50,400),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),1)
                elif recognize_result == 0:
                    print('The operation failed, please try again')
					
                else:
                    dict_candidate[recognize_result]+=1
        person = 0
        for name in names:
            person += 1
            cv2.putText(processed,name+' appears: '+str(dict_candidate[name])+' times',(50,50*person),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),1)
        
        
        if savevideo:
            videoWriter.write(processed)
        cv2.imshow('', processed)
    
        choice = cv2.waitKey(1)
        if choice&0xFF == 27: break
    
    if savevideo:
        videoWriter.release()
    camera.release()
    cv2.destroyAllWindows()


#-------------------main-----------------------

names, paths = face.scan_images('candidate-faces')
count = np.zeros(len(names))
dict_candidate = dict(zip(names,count))
descriptors = face.get_descriptors(paths)
camera_realtime(0,True)