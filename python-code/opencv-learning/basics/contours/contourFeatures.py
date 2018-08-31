#coding:utf-8
import cv2
import numpy as np

img = cv2.imread('datas/binary.jpg', 1)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV)

image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]
M = cv2.moments(cnt)

print(M)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
print('centroid %d %d' % (cx, cy))
print('contour length', cv2.arcLength(cnt, True))
print('contour area', cv2.contourArea(cnt))
x, y, w, h = cv2.boundingRect(cnt)

# Bounding rectangle
cv2.rectangle(img, (x, y), (x + w, y + h), 0, 1)

# Minimal area bounding rectangle
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

cv2.circle(img, (cx, cy), 3, (0, 255, 0), -1)
cv2.imshow('center', img)
cv2.waitKey(0)
cv2.destroyAllWindows()