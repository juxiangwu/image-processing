"""
@introduction:
------------------------------------------------------------------------------
    Introduction
    ============
    This module is used to experiment and the test the feature generation
    and data representation for SVM.
------------------------------------------------------------------------------
@usage:
------------------------------------------------------------------------------
    Usage
    =====
    Run the module as a command line option for python interpreter.
    -> python py_experiment_features.py
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
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import os
import cv2
import dlib
import math
import numpy as np
import datetime as dt

# ---------------------------------------------------------------------------
# Module info
# ---------------------------------------------------------------------------

__author__ = "Mani Kumar D A - 2017, Paul van Gent - 2016"
__version__ = "2.1, 24/06/2017"
__license__ = "GNU GPL v3"
__copyright__ = "Mani Kumar D A"

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

dirsep = os.sep  # directory path separator
curdir = os.curdir  # Relative current directory i.e. '.'
cwdpath = os.getcwd()  # current working directory full path name

video_capture = cv2.VideoCapture(0)  # Webcam object
claheObject = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

frontalFaceDetector = dlib.get_frontal_face_detector()
facialShapePredictor = dlib.shape_predictor(
    "..{0}input{1}shape_predictor_68_face_landmarks.dat".format(
        dirsep, dirsep))
RAD2DEG_CVT_FACTOR = 180 / math.pi  # Constant to convert radians to degrees.

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def main():
    """
    Main function - start of the program.
    """
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray)
        detections = frontalFaceDetector(clahe_image, 1)
        for d in detections:  # For each detected face
            shape = facialShapePredictor(clahe_image, d)  # Get coordinates
            xCoordinatesList = []
            yCoordinatesList = []
            # Store the X and Y coordinates of landmark points in two lists
            for i in range(0, 68):
                xCoordinatesList.append(shape.part(i).x)
                yCoordinatesList.append(shape.part(i).y)
            # Get the mean of both axes to determine centre of gravity
            xCoordMean = np.mean(xCoordinatesList)
            yCoordMean = np.mean(yCoordinatesList)
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
            # For each 68 landmark points.
            for i in range(1, 68):
                '''
                cv2.arrowedLine(frame, (int(xCoordMean), int(yCoordMean)),
                                (xCoordinatesList[i], yCoordinatesList[i]),
                                (0, 0, 255), thickness=1, line_type=4,
                                shift=0, tipLength=0.05)
                '''
                '''
                cv2.circle(frame, (xCoordinatesList[i], yCoordinatesList[i]),
                           1, (0, 0, 255), thickness=2)
                cv2.putText(frame, "{}".format(i), (xCoordinatesList[i],
                                                    yCoordinatesList[i]),
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.3,
                                                    (0, 255, 0),
                                                    thickness=1)
                '''
                '''
                # For each point, draw circle with thickness = 2 on the
                # original frame
                if i == 27 or i == 30:
                    cv2.circle(frame, (xCoordinatesList[i],
                                       yCoordinatesList[i]),
                                       1,
                                       (0, 255, 0),
                                       thickness=2)
                    cv2.putText(frame,
                                "P{}".format(i),
                                (xCoordinatesList[i], yCoordinatesList[i]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                (0, 255, 0),
                                thickness=1)
                else:
                    # pass
                    cv2.circle(frame,
                               (xCoordinatesList[i], yCoordinatesList[i]),
                               1,
                               (0, 0, 255),
                               thickness=2)
                '''
            # For mean coordinates.
            cv2.circle(frame,
                       (int(xCoordMean), int(yCoordMean)),
                       1, (255, 0, 0), thickness=2)
            cv2.putText(frame, "mean({}, {})".format(int(xCoordMean),
                                                     int(yCoordMean)),
                        (int(xCoordMean), int(yCoordMean)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (255, 0, 0), thickness=1)
        cv2.imshow("image", frame)  # Display the frame
        # Save the frame when the user presses 's'
        if cv2.waitKey(1) & 0xFF == ord('s'):
            img_name = "..{0}img_samples{1}img_cap_{2}.jpg".format(dirsep,
                dirsep, dt.datetime.today().strftime("%Y%m%d_%H%M%S"))
            cv2.imwrite(img_name, frame)
        # Exit program when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break