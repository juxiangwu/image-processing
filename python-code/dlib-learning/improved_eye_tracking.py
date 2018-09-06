import cv2
import imutils as im
import dlib

##==========start the video==================
cap = cv2.VideoCapture('../resources/vidoes/eye/ankit-6-10-2017_1200.h264')

##==============create objects for the detectors and predictors================
detector = dlib.simple_object_detector("../resources/models/dlib/eye-track/pupil.svm")
predictor = dlib.shape_predictor("../resources/models/dlib/eye-track/pupilPredictor.dat")
Goteye = dlib.simple_object_detector("../resources/models/dlib/eye-track/GotEye.svm")

##==============declare universal constants===================================
eyeDetected = False
buffer = 0
##==============start the video=============================
while True:
    ret, frame = cap.read()
    frame = im.resize(frame,640,480,cv2.INTER_AREA)
    output = frame.copy()
    rects = Goteye(frame)


    ## search for the eye in the frame---------------
    if len(rects)>0 and eyeDetected==False :
        ##------------if detected, confirm it in next subsequent frames----------------
        buffer += 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Hold on, stabilizing it!', (0, 30), font, 1, (200, 255, 155), 2, cv2.LINE_AA)

        if buffer==20:
        ##----once confirmed proceed---------------
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Got it! continue', (0, 50), font, 1, (200, 255, 155), 2, cv2.LINE_AA)
            for k, d in enumerate(rects):
                cv2.rectangle(output, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 255))
            eyeDetected = True
            frame = frame[d.left():d.right(), d.top(): d.bottom()]

        for k, d in enumerate(rects):


            cv2.rectangle(output,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,255))
            shape = predictor(output, d)
            print("left: {}, right: {} , top: {}".format(shape.part(0),shape.part(1),shape.part(2)))
            # Draw the face landmarks on the screen.
    if eyeDetected:
        ##-------destroy the standby window------
        cv2.destroyWindow("waiting to confirm...")

        dets = detector(frame,1)
        ##---------detect pupil int the eye frame-------------------
        for k, d in enumerate(dets):

            if k==0:
                cv2.rectangle(output,(d.left(),d.top()),(d.right(),d.bottom()),(0,0,255))
                x = int((d.left() +d.right())/2)
                y = int((d.top()+d.bottom())/2)
                print("coordinates of eye:",(x,y))
                cv2.circle(output,(x,y),2,(0,255,255),2)
                pupil = frame[d.top():d.bottom(), d.left():d.right()]
                shape = predictor(pupil, d)
                ##--------predict shape of the pupil-------------
                print("left: {}, right: {} , top: {}".format(shape.part(0), shape.part(1), shape.part(2)))

                left =  shape.part(0)
                right = shape.part(1)
                top = shape.part(2)
                lx = left.x
                ly = left.y
                rx = right.x
                ry = right.y
                tx = top.x
                ty = top.y
                cv2.circle(output,(lx,ly),2,(255,0,0),1)
                cv2.circle(output, (rx,ry), 2, (0, 255, 0), 1)
                cv2.circle(output, (tx,ty), 2, (0, 0, 255), 1)
                # Draw the face landmarks on the screen.

        if len(dets)==0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(output, 'Please hold on a little longer...', (0, 50), font, 1, (200, 100, 155), 2, cv2.LINE_AA)


        # cv2.imshow("out",edge)
        cv2.imshow("Perimetry test in process.. ",output)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Please wait, waiting to confirm...', (0, 130), font, 1, (200, 255, 155), 2, cv2.LINE_AA)
        cv2.imshow("waiting to confirm...", frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


cap.release()
cv2.destroyAllWindows()