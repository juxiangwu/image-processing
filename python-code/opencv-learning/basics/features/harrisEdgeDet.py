import cv2
import numpy as np

img = cv2.imread('datas/geometricFlower.jpg', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# For Harris corner det. float 32 type is needed.
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 5, 5, 0.1)

img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow('corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()