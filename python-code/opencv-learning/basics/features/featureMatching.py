import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('datas/book.jpg')
img2 = cv2.imread('datas/bookInContext.jpg')

orb = cv2.ORB_create(500, 1.1, 16, 40, patchSize=40)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
img3 = np.hstack((img1, img2))
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)
cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], img3, (0, 255, 0), flags=2)

cv2.imshow('img3', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()