#coding:utf-8
import cv2
from PyQt5.QtGui import QPixmap,QImage
def load_image(fname,isGray=False):
    if isGray:
        return cv2.imread(fname,0)
    else:
        return cv2.imread(fname)

def array2pixmap(image):
    rows,cols = image.shape[0],image.shape[1]
    print('array2pixmap:shape=',image.shape)
    rgb = image
    if len(image.shape) < 3:
        rgb = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    rgb =cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
    # print(rgb.data)
    try:
        qimg = QImage(rgb.data,cols,rows,cols*3,QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)
    except Exception as ex:
        print(ex)