import cv2
import numpy as np
import os
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.externals import joblib
#import scikit-image.skimage.data
#, color, exposure


array=np.array([])	#empty array for storing all the features
h = 128			#height of the image
w = 64			#width of the image
hog=cv2.HOGDescriptor()
img=cv2.imread('datas/plane.jpg',0)
img1=cv2.imread('datas/plane_part.jpg',0)

img=cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)	#resize images
img1=cv2.resize(img1, (w, h), interpolation=cv2.INTER_CUBIC)
for i in range(0,2):
	if i==0:
		h=hog.compute(img,winStride=(64,128),padding=(0,0))	#storing HOG features as column vector
		#hog_image_rescaled = exposure.rescale_intensity(h, in_range=(0, 0.02))
		#plt.figure(1, figsize=(3, 3))
		#plt.imshow(h,cmap=plt.cm.gray)
		#plt.show()
		#print len(h)
		h_trans=h.transpose()	#transposing the column vector
	
		array=np.vstack(h_trans)	#appending it to the array
		#print "HOG features of label 1"
		#print array
		
	else:
		h=hog.compute(img1,winStride=(64,128),padding=(0,0))	#storing HOG features as column vector
		
		#hog_image_rescaled = exposure.rescale_intensity(h, in_range=(0, 0.02))
		#plt.figure(1, figsize=(3, 3))
		#plt.imshow(h,cmap=plt.cm.gray)
		#plt.show()
		#print len(h)
		h_trans=h.transpose()	#transposing the column vector
	
		array=np.vstack((array,h_trans))	#appending it to the array
		#print "HOG features of label 1 & 2"
		#print (array)	

#print (array.shape)


label=[1,4]

clf=SVC(gamma=0.001,C=10)

clf.fit(array,label)

#ypred = clf.predict()
joblib.dump(clf, "temp/hog.pkl", compress=3)