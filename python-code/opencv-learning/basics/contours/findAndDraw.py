#coding:utf-8
import cv2
import numpy as np

img = cv2.imread('datas/binary.jpg', 1)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY)
imgGray, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, 1, (255, 0, 0), 3)

cv2.imshow('contour', img)
cv2.waitKey(0)
cv2.destroyAllWindows()