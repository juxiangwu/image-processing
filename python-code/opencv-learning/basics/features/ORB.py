import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('datas/geometricFlower.jpg', 1)
img2 = img.copy()

orb = cv2.ORB_create()

kp = orb.detect(img, None)
kp, des = orb.compute(img, kp)
cv2.drawKeypoints(img, kp, img2, (0, 255, 0), flags=0)
cv2.imshow('asdf', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()