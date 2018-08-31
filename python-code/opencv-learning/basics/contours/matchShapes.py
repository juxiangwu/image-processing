#coding:utf-8
import cv2
import numpy as np

imgToFind = cv2.imread('datas/toFind.jpg', 0)
imgFromFind = cv2.imread('datas/fromFind.jpg', 0)

ret, threshTo = cv2.threshold(imgToFind, 127, 255, cv2.THRESH_BINARY_INV)
ret, threshFrom = cv2.threshold(imgFromFind, 127, 255, cv2.THRESH_BINARY_INV)

cntTo = cv2.findContours(threshTo, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cntFrom = cv2.findContours(threshFrom, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cntTo = cntTo[0]
cntFrom = cntFrom[0]

diff = cv2.matchShapes(cntTo, cntFrom, 1, 0.0)

print(diff)