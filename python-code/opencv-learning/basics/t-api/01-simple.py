#coding:utf-8
import cv2

mat = cv2.imread('datas/bird.jpg')
umat = cv2.UMat(mat)

dst = cv2.cvtColor(umat,cv2.COLOR_BGR2GRAY)

cv2.imshow('src',mat)
cv2.imshow('dst',dst)

cv2.waitKey()
cv2.destroyAllWindows()