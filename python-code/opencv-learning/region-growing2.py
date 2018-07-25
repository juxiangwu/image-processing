#coding:utf-8

from scipy.spatial import distance
from pylab import *
import numpy as np 
from pylab import *
import matplotlib.pyplot as plt
import cv2


#Function that returns the segmented image
def regionGrowing(I,click):
    
    shap= np.shape(I)
    seed1= [int(click[0][0]),int(click[0][1])]# pixel indices of the mouse click on the image
    sample= np.zeros(shap) # arbitary pixel initialization which would be used for finding the Euclidean Distance
    
    threshold=5 
    imSegment= (np.ones(shap)) * 255   #Set the gray scale image 
    imSegment[seed1[0],seed1[1]] = 255 #Initialize a matrix with 
    
    seed = [[232,173]] # a starting point for the region growing
    
    init_seed=[] 
    count = 0
    # Loop that appends the seed to the initial starting point for all of the pixels
    
    
    for i in range(shap[0] * shap[1]): 
                try:
                    init_seed= seed[count]
                    count = count+1
            
                    x= init_seed[0] 
                    y= init_seed[1]
                    
                    if (x<shap[0]-1) and (y<shap[1]-1):
                        neighbour=[x-1,y],[x+1,y],[x,y-1],[x,y+1]# 4 point connected for comparing with the neighbouring pixels
                    for j in neighbour:
                        if sample[j[0],j[1]]==0:
                            if distance.euclidean(I[x,y],I[j[0],j[1]])> threshold : 
                                imSegment[j[0],j[1]] =0 # set the segmented image pixels to black
                            sample[j[0],j[1]]=255 # set the sample to white
                            seed.append([j[0],j[1]])
                            
                except IndexError:
                        print "Please re-run the program and choose another point on the image"
                        break
                

                            
    return imSegment # output image is returned 
              
                       
 
I = cv2.imread('input1.jpg', 0)
# open the image for selecting the point for image growing from the user
figure()
plt.imshow(I, cmap = cm.gray)  
plt.show()


#Selecting the seed array for initialization using Mouse Click on the input image
click = ginput(2)# mouse click input
plt.close()

R = regionGrowing(I,click)
plt.imshow(R, cmap = cm.gray) # output the segmented image
plt.show()

Out = cv2.imread('out1.jpg',0)

#F- score calculation
TP = TN = FP = FN= 0.0
Shape = np.shape(Out)
for i in range(Shape[0]):
    for j in range(Shape[1]):
        if R[i,j] == 0 and Out[i,j] == 0: #True Positive
            TP = TP + 1
        if R[i,j] == 255 and Out[i,j] == 0:# False Negative
            FN = FN + 1
        if R[i,j] == 255 and Out[i,j] == 255:# True Negative
            TN = TN + 1
        if R[i,j] == 0 and Out[i,j] == 255:# False Positive
            FP = FP + 1
        
F1_score = 2*TP/((2*TP)+FP+FN) #F1 score 
print 'F1 score is =',F1_score
