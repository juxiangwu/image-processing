#coding:utf-8
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow('frame',frame)
    fgmask = fgbg.apply(frame)
    cv2.imshow('fg', fgmask)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()