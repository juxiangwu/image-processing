"""
@introduction:
------------------------------------------------------------------------------
    Introduction
    ============
    This module is used to load and test the OpenCV SVM model.
    SVM is short for Support Vector Machine, a Machine Learning algorithm
    used to classify data but can also be used for regression. I am using it
    to classify the different states [classes in ML terms] of human face.
    Facial landmarks are extracted using dlib's shape predictor with a
    pre-trained shape predictor model. These landmarks are further processed
    as mentioned in the tutorial from Paul Van Gent and fed to the svm.train
    method.
    I have modified the code used in the tutorial and experimented with
    simpler and easy to read code. For further description please follow the
    tutorial by Paul mentioned in the link in README.md of this repo.
------------------------------------------------------------------------------
@usage:
------------------------------------------------------------------------------
    Usage
    =====
    Run the module as a command line option for python interpreter.
    -> python py_test_trained_svm_model.py
------------------------------------------------------------------------------
@purpose:
------------------------------------------------------------------------------
    To understand the data, feature generation and representation, SVM model
    training and SVM prediction.
------------------------------------------------------------------------------
@based_on:
------------------------------------------------------------------------------
    <a href="http://www.paulvangent.com/">
       Emotion Recognition using Facial Landmarks, Python, DLib and OpenCV
    </a>
------------------------------------------------------------------------------
@applications:
------------------------------------------------------------------------------
    1. Image information extraction.
    2. Understanding the machine learning aspects.
    3. Data extraction and representation.
------------------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import cv2
import math
import dlib
import numpy as np
import datetime as dt

# ---------------------------------------------------------------------------
# Module info
# ---------------------------------------------------------------------------

__author__    = "Mani Kumar D A - 2017, Paul van Gent - 2016"
__version__   = "2.1, 24/06/2017"
__license__   = "GNU GPL v3"
__copyright__ = "Mani Kumar D A"

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

dirsep             = os.sep  # directory path separator
curdir             = os.curdir  # Relative current directory i.e. '.'
cwdpath            = os.getcwd()  # current working directory full path name

emotionsList       = ["anger", "contempt",
                      "happy", "neutral",
                      "sadness", "surprise"]  # Human facial expression states

''' -> Yet to add `pout`, `disgust` and `fear` are excluded '''

frontalFaceDetector  = dlib.get_frontal_face_detector()
facialShapePredictor = dlib.shape_predictor(
    "..{0}input{1}shape_predictor_68_face_landmarks.dat".format(
        dirsep, dirsep))

RAD2DEG_CVT_FACTOR = 180 / math.pi  # Constant to convert radians to degrees.
# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_landmarks(claheImage):
    """ Get landmarks from detected faces. """
    detectedFaces = frontalFaceDetector(claheImage, 1)

    for detectedFace in detectedFaces:
        xCoordinatesList   = []
        yCoordinatesList   = []
        landmarkVectorList = []
        facialShape = facialShapePredictor(claheImage, detectedFace)
        # Store the X and Y coordinates of landmark points in two lists
        for i in range(0, 68):
            xCoordinatesList.append(facialShape.part(i).x)
            yCoordinatesList.append(facialShape.part(i).y)
        # Determine centre of gravity point by calculating the mean of
        # all points.
        xCoordMean = np.mean(xCoordinatesList)
        yCoordMean = np.mean(yCoordinatesList)
        # Point 27 represents the top of the nose bridge and point 30
        # represents the tip of the nose - follow the PVG tutorial for
        # more details on this step.
        if xCoordinatesList[27] == xCoordinatesList[30]:
            noseBridgeAngleOffset = 0
        else:
            radians1 = math.atan(
                (yCoordinatesList[27] - yCoordinatesList[30]) /
                (xCoordinatesList[27] - xCoordinatesList[30]))
            noseBridgeAngleOffset = int(radians1 * RAD2DEG_CVT_FACTOR)
        if noseBridgeAngleOffset < 0:
            noseBridgeAngleOffset += 90
        else:
            noseBridgeAngleOffset -= 90
        for xcoord, ycoord in zip(xCoordinatesList, yCoordinatesList):
            xyCoordArray = np.asarray((ycoord, xcoord))
            xyCoordMeanArray = np.asarray((yCoordMean, xCoordMean))
            pointDistance = np.linalg.norm(xyCoordArray - xyCoordMeanArray)
            denom = (xcoord - xCoordMean)  # Prevent divide by zero error.
            if denom == 0:
                radians2 = 1.5708  # 90 deg = 1.5708 rads
            else:
                radians2 = math.atan((ycoord - yCoordMean) / denom)
            pointAngle = (radians2 * RAD2DEG_CVT_FACTOR) - noseBridgeAngleOffset
            landmarkVectorList.append(pointDistance)
            landmarkVectorList.append(pointAngle)

    if len(detectedFaces) < 1:
        landmarkVectorList = "No face detected!"
    return landmarkVectorList


def main():
    """
    Main function - start of the program.
    """
    print "\n main() - Enter"
    svm = cv2.SVM()
    frameText = ""
    claheObject = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # ------------------------- Loading opencv SVM ------------------------
    print "\n#################### Loading opencv SVM ####################\n"
    svm.load("..{0}input{1}cv2_svm_6_states.yml".format(dirsep, dirsep))
    print "Loading opencv SVM model from file - Completed."
    # ------------------------- Start Webcam ------------------------------
    vidCap = cv2.VideoCapture(0)  # Webcam object
    '''
    fourcc = cv2.cv.CV_FOURCC('F', 'M', 'P', '4')  # Video writer object.
    '''
    vidOutPath = "..{0}video_out{1}vid_cap_{2}.avi".format(
            dirsep, dirsep, 
            dt.datetime.today().strftime("%Y%m%d_%H%M%S"))
    # vidWriter = cv2.VideoWriter(vidOutPath, fourcc, 20.0, (640, 480))
    vidWriter = cv2.VideoWriter(vidOutPath, -1, 5.0, (640,480))
    print "\n##################### Starting Webcam ######################\n"

    while (vidCap.isOpened()):
        ret, frame = vidCap.read()
        if not ret:
            print "Frame not read, video captured returned {}.".format(ret)
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        claheImage = claheObject.apply(gray)
        landmarkVectorList = get_landmarks(claheImage)
 
        if landmarkVectorList == "No face detected!":
            frameText = "No face detected!"
            cv2.putText(frame, "{}".format(frameText), (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), thickness=2)
        else:
            # -------------------- Testing opencv SVM -----------------------
            # Testing data must be float32 matrix for the opencv svm.
            npArrTestData = np.float32(landmarkVectorList)
            result = svm.predict(npArrTestData)
            # -------------------- Print result -----------------------------
            frameText = emotionsList[int(result)]
            cv2.putText(frame, "You are {}.".format(frameText),
                        (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), thickness=2)
        
        vidWriter.write(frame)  # Write video frame to file.
        cv2.imshow("Frame", frame)  # Display the frame
        # Save the frame when the user presses 's'
        if cv2.waitKey(1) & 0xFF == ord('s'):
            img_name = "..{0}img_samples{1}img_cap_{2}.jpg".format(
            dirsep, dirsep, 
            dt.datetime.today().strftime("%Y%m%d_%H%M%S"))
            cv2.imwrite(img_name, frame)
        
        # Exit program when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    vidCap.release()
    vidWriter.release()
    cv2.destroyAllWindows()
    print "\n main() - Exit"


if __name__ == '__main__':
    main()
