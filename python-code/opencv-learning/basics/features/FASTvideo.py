import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fast = cv2.FastFeatureDetector_create()

while True:
    ret, frame = cap.read()
    kp = fast.detect(frame, None)
    cv2.drawKeypoints(frame, kp, frame, (0, 255, 0))
    cv2.imshow('kp', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()