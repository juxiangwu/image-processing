""" Simple Script to detect Red eye and Auto Correct it
Here we assume that image is a face potrait ,
We use the standard OpenCV Haar detector (haarcascade_eye.xml) for eyes for finding the eyes in the image .
"""

import cv2
import numpy as np

#read image
img = cv2.imread("datas/red_eye_wiki.jpg") # pass the name of image to be read

outImage = img.copy()
# Load HAAR cascade
eyesCascade = cv2.CascadeClassifier("datas/cvdata/haarcascades/haarcascade_eye.xml")

#------------ Detect Eyes ------------#
# eyeRects contain bounding rectangle of all detected eyes

eyeRects = eyesCascade.detectMultiScale(img , 1.1, 5 )

#Iterate over all eyes to remove red eye defect

for x,y,w,h in eyeRects:

    #Crop the eye region
    eyeImage = img [y:y+h , x:x+w]

    #split the images into 3 channels
    b, g ,r = cv2.split(eyeImage)

    # Add blue and green channels
    bg = cv2.add(b,g)

    #threshold the mask based on red color and combination ogf blue and gree color
    mask  = ( (r>(bg-20)) & (r>80) ).astype(np.uint8)*255

    #Some extra region may also get detected , we find the largest region
    #find all contours
    contours, _ = cv2. findContours(mask.copy() ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )  # It return contours and Hierarchy

    #find contour with max Area
    maxArea = 0
    maxCont = None
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > maxArea:
            maxArea = area
            maxCont = cont
    mask = mask * 0  # Reset the mask image to complete black image
    # draw the biggest contour on mask
    cv2.drawContours(mask , [maxCont],0 ,(255),-1 )
    #Close the holes to make a smooth region
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_DILATE,(5,5)) )
    mask = cv2.dilate(mask , (3,3) ,iterations=3)

    #--The information of only red color is lost ,
    # So we fill the mean of blue and green color in all three channels(BGR) to maintain the texture
    mean  = bg /2

    # Fill this black mean value to masked image
    mean = cv2.bitwise_and(mean , mask )  # mask the mean image
    mean  = cv2.cvtColor(mean ,cv2.COLOR_GRAY2BGR ) # convert mean to 3 channel
    mask = cv2.cvtColor(mask ,cv2.COLOR_GRAY2BGR )  #convert mask to 3 channel
    eye = cv2.bitwise_and(~mask,eyeImage)+mean           #Copy the mean color to masked region to color image
    outImage [y:y+h , x:x+w] = eye

# Stack both input and output image horizontally

result = np.hstack((img,outImage))

#Display the Result
cv2.imshow("RedEyeCorrection" , result )
cv2.waitKey()  # Wait for a keyPress
cv2.destroyAllWindows()