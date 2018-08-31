"""
@introduction:
------------------------------------------------------------------------------
    Introduction
    ============
    This module is used to train and save the OpenCV SVM model.
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
    -> python py_train_save_svm_model.py
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
import glob
import math
import dlib
import random
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


def get_files(emotion):
    """
    Get all files list, randomly shuffle it and split into 80/20.
    -> 80% for training data
    -> 20% for testing data
    @params:
       emotion : str => emotion folder name.
    @return:
       training, prediction : list of str => list of the file names
                              i.e. images for the emotion.
    """
    print "\n get_files({0}) - Enter".format(emotion)
    files = glob.glob("..{0}dataset{1}{2}{3}*".format(dirsep, dirsep,
                                                      emotion, dirsep))
    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]  # get first 80% of file list
    prediction = files[-int(len(files) * 0.2):]  # get last 20% of file list
    print "\n get_files({0}) - Exit".format(emotion)
    return training, prediction


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


def make_sets():
    print "\n make_sets() - Enter"
    training_data     = []
    training_labels   = []
    prediction_data   = []
    prediction_labels = []    
    claheObject = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for emotion in emotionsList:
        training, prediction = get_files(emotion)
        for item in training:
            image = cv2.imread(item)  # read image
            gray = cv2.cvtColor(  # convert to grayscale
                        image, cv2.COLOR_BGR2GRAY)
            clahe_image = claheObject.apply(gray)
            landmarkVectorList = get_landmarks(clahe_image)
            if landmarkVectorList == "No face detected!":
                pass
            else:
                training_data.append(landmarkVectorList)
                training_labels.append(emotionsList.index(emotion))
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = claheObject.apply(gray)
            landmarkVectorList = get_landmarks(clahe_image)
            if landmarkVectorList == "No face detected!":
                pass
            else:
                prediction_data.append(landmarkVectorList)
                prediction_labels.append(emotionsList.index(emotion))

    print "\n make_sets() - Exit"
    return training_data, training_labels, prediction_data, prediction_labels


def main():
    """
    Main function - start of the program.
    """
    print "\n main() - Enter"
    svm = cv2.SVM()
    svm_params = dict(kernel_type=cv2.SVM_LINEAR,
                      svm_type=cv2.SVM_C_SVC,
                      C=2.67,
                      gamma=5.383)
    maxRuns = 1
    runCount = 0
    predictionAccuracyList = [0] * maxRuns

    for runCount in range(0, maxRuns):
        print "\n\t\t<--- Making sets {0} --->".format(runCount)
        training_data, training_labels, prediction_data, prediction_labels =\
            make_sets()
        # ---------------------- Training opencv SVM ----------------------
        print "\n################## Training opencv SVM ##################\n"
        # Training data must be float32 matrix for the opencv svm.
        npArrTrainData = np.float32(training_data)
        npArrTrainLabels = np.float32(training_labels)
        print "npArrTrainData.shape = {0}.".format(npArrTrainData.shape)
        print "npArrTrainLabels.shape = {0}.".format(npArrTrainLabels.shape)
        print "Training opencv SVM linear {0} - Started.".format(runCount)
        svm.train(npArrTrainData, npArrTrainLabels, params=svm_params)
        print "Training opencv SVM linear {0} - Completed.".format(runCount)

        # Save opencv SVM trained model.
        svm.save("..{0}input{1}cv2_svm_6_states.yml".format(dirsep,
                                                            dirsep))
        print "\nSaving opencv SVM model to file - Completed."

        # ------------------- Testing opencv SVM --------------------------
        print "\n################## Testing opencv SVM ###################\n"
        # Testing data must be float32 matrix for the opencv svm.
        npArrTestData = np.float32(prediction_data)
        npArrTestLabels = np.float32(prediction_labels)
        print "npArrTestData.shape = {0}.".format(npArrTestData.shape)
        print "npArrTestLabels.shape = {0}.".format(npArrTestLabels.shape)
        print "Testing opencv SVM linear {0} - Started.".format(runCount)
        results = svm.predict_all(npArrTestData).reshape((-1,))
        print "Testing opencv SVM linear {0} - Completed.".format(runCount)
        print "\n\t-> type(npArrTestLabels) = {}".format(
                                                  type(npArrTestLabels))
        print "\t-> type(npArrTestLabels[0]) = {}".format(
                                                  type(npArrTestLabels[0]))
        print "\t-> npArrTestLabels.size = {}".format(npArrTestLabels.size)
        print "\n\t-> type(results) = {}".format(type(results))
        print "\t-> type(results[0]) = {}".format(type(results[0]))
        print "\t-> results.size = {}, results.shape = {}".format(
                                                results.size, results.shape)
        # ------------------- Check Accuracy ---------------------------------
        print "\n################## Check Accuracy #######################\n"
        mask = results == npArrTestLabels
        correct = np.count_nonzero(mask)
        print "\t-> type(mask) = {}".format(type(mask))
        print "\t-> type(mask[0]) = {}".format(type(mask[0]))
        print "\t-> mask.size = {}, mask.shape = {}".format(mask.size,
                                                            mask.shape)
        pred_accur = correct * 100.0 / results.size
        print "\nPrediction accuracy = %{0}.\n".format(pred_accur)
        print "--------------------------------------------------------------"
        predictionAccuracyList[runCount] = pred_accur
        # predictionAccuracyList.append(pred_accur)
    # Get the mean accuracy of the i runs
    print "Mean value of predict accuracy in {0} runs: {1:.4f}".format(
        maxRuns, sum(predictionAccuracyList) / len(predictionAccuracyList))
    print "\n main() - Exit"


if __name__ == '__main__':
    main()