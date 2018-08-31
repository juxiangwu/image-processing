import cv2
import numpy as np
import time

# Gets the average color of an image and displays the average color in a window
t1 = time.time()

img = cv2.imread("datas/red-black-1920x1080-wallpaper367453.jpg", 1)
average = cv2.mean(img)
cv2.imshow("original", img)

# Generating average colored image
avImg = np.ones((256, 256, 3), np.uint8)
x = np.array(list(average[0:3]))
x = x.astype(np.uint8)
# print x
avImg = avImg[:,:] * x
cv2.imshow("average color",avImg)
t2 = time.time()

# print t2 -t1
cv2.waitKey(0)