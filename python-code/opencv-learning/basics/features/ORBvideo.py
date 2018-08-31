import cv2
import numpy as np

cap = cv2.VideoCapture(0)
orb = cv2.ORB_create(500)

while True:
    ret, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = orb.detect(imgGray, None)
    cv2.drawKeypoints(imgGray, kp, img, (0, 255, 0))
    cv2.imshow('kp', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.relsease()
cv2.destroyAllWindows()