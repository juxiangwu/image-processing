import cv2
import numpy as np

img = cv2.imread('datas/geometricFlower.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fast = cv2.FastFeatureDetector_create()

kp = fast.detect(gray, None)
img2 = img.copy()
cv2.drawKeypoints(img, kp, img2, (255, 0, 0))

cv2.imshow('sadf', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()