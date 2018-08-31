import cv2
import numpy as np

img = cv2.imread('datas/chessboard.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect 100 corners with min 0.25 quality and 10px distance between them.
corners = cv2.goodFeaturesToTrack(gray, 100, 0.25, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

cv2.imshow('corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()