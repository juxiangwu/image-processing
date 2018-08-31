import cv2
import numpy as np
import time

img = cv2.imread('datas/book2.jpg')
cap = cv2.VideoCapture(0)

orb = cv2.ORB_create(2500, 1.2, 11)
kp1, des1 = orb.detectAndCompute(img, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)

while True:
    t1 = time.time()
    ret, frame = cap.read()

    imgOut = np.hstack((img, frame))

    kp2, des2 = orb.detectAndCompute(frame, None)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    cv2.drawMatches(img, kp1, frame, kp2, matches[:100],
                    imgOut, (0, 255, 0), flags=2)
    cv2.imshow('matches', imgOut)

    t2 = time.time()
    print(t2 - t1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()