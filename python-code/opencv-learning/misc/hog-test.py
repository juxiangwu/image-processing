import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np

h = 128
w = 64

# Load the classifier
clf = joblib.load("temp/hog.pkl")

# Read the input image 
im = cv2.imread("datas/plane_part_2.jpg")

img=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
img=cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)

roi_hog_fd = hog(img, orientations=105, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)

nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
print(nbr)