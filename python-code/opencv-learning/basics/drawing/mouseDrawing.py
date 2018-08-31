import cv2
import numpy as np

def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 100, (0, 255, 0), 5)

img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow("image")
cv2.setMouseCallback("image",draw_circle)

while True:
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()