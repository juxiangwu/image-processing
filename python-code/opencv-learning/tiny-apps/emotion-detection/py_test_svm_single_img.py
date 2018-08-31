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
    -> python py_test_svm_single_img.py
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

RAD2DEG_CVT_FACTOR = 180 / math.pi  # Constant to convert radians to degrees.
emotionsList       = ["anger", "contempt",
                      "happy", "neutral",
                      "sadness", "surprise"]  # Human facial expression states

''' -> Yet to add `pout`, `disgust` and `fear` are excluded '''

frontalFaceDetector  = dlib.get_frontal_face_detector()
facialShapePredictor = dlib.shape_predictor(
    "..{0}input{1}shape_predictor_68_face_landmarks.dat".format(
        dirsep, dirsep))

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def get_landmarks(claheImage):
    detectedFaces = frontalFaceDetector(claheImage, 1)

    # For all detected face instances extract the features
    for detectedFace in detectedFaces:
        xCoordinatesList = []
        yCoordinatesList = []
        landmarkVectorList = []
        # Draw Facial Landmarks with the predictor class
        facialShape = facialShapePredictor(claheImage, detectedFace)
        # Store the X and Y coordinates of landmark points in two lists
        for i in range(0, 68):
            xCoordinatesList.append(facialShape.part(i).x)
            yCoordinatesList.append(facialShape.part(i).y)

        # Get the mean of both axes to determine centre of gravity
        xCoordMean = np.mean(xCoordinatesList)
        yCoordMean = np.mean(yCoordinatesList)
        # If x-coordinates of the set are the same, the angle is 0,
        # catch to prevent 'divide by 0' error in the function
        if xCoordinatesList[27] == xCoordinatesList[30]:
            noseBridgeAngleOffset = 0
        else:
            radians1 = math.atan(
                (yCoordinatesList[27] - yCoordinatesList[30]) /
                (xCoordinatesList[27] - xCoordinatesList[30]))
            # since degrees = radians * RAD2DEG_CVT_FACTOR
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
        landmarkVectorList = "error"
    return landmarkVectorList


# Set the classifier as a opencv svm with SVM_LINEAR kernel
maxRuns = 100
runCount = 0
svm = cv2.SVM()
predictionAccuracyList = [0] * maxRuns
claheObject = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

for runCount in range(0, maxRuns):
    # Get a sample for prediction
    fileName = raw_input("Enter file name: ")
    if fileName == "quit" or fileName == "q":
        print "Quitting the application!"
        break
    else:
        print "File name is: {}".format(fileName)
    # Get landmark features from the image.
    image = cv2.imread(fileName)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_image = claheObject.apply(gray)
    landmarkVectorList = get_landmarks(clahe_image)
    if landmarkVectorList == "error":
        print "Feature extraction returns error!"
        continue  # Go to the next run.

    print "\n#################### Loading opencv SVM ####################\n"
    # Load opencv SVM trained model.
    svm.load("..{0}input{1}cv2_svm_6_states.yml".format(dirsep,
                                                              dirsep))
    print "Loading opencv SVM model from file - Completed."
    print "\n#################### Testing opencv SVM ####################\n"
    # Testing data must be float32 matrix for the opencv svm.
    npArrTestData = np.float32(landmarkVectorList)
    print "npArrTestData.shape = {0}.".format(npArrTestData.shape)
    print "Testing opencv SVM linear {0} - Started.".format(runCount)
    # results = svm.predict_all(npArrTestData).reshape((-1,))
    result = svm.predict(npArrTestData)
    print "Testing opencv SVM linear {0} - Completed.".format(runCount)
    print "\n#################### Result ####################\n"
    print "result: emotionsList[{0}] = {1}".format(result,
                                                   emotionsList[int(result)])
    predictionAccuracyList.append(result)
    print "---------------------------------------------------------------"

# Get the mean accuracy of the i runs
print "Mean value of predict accuracy in {0} runs: {1:.4f}".format(
    maxRuns, np.mean(predictionAccuracyList))
# sum(predictionAccuracyList) / len(predictionAccuracyList)