#coding:utf-8
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

rectangleNEG = cv2.imread('datas/circle2.jpg', 0)
ret, rectangle = cv2.threshold(rectangleNEG, 127, 255, cv2.THRESH_BINARY_INV)

recC, contours, hierarchy = cv2.findContours(rectangle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
recCont = contours[0]

while True:
    ret, frame = cap.read()
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #canny = cv2.Canny(frameGray, 70, 150)
    canny = cv2.adaptiveThreshold(frameGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 6)
    cv2.imshow('canny', canny)
    grayC, contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 1)
        epsilon = 0.000001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if cv2.matchShapes(approx, recCont, 2, 0.0) < 0.1:
            if cv2.contourArea(approx) > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('stream', frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()