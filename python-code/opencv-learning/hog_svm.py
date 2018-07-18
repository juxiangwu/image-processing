# -*- coding: utf-8 -*-
import cv2
import sys
import os

def getFilePath(path, filter=''):
    file_list = []
    for (root, dirs, files) in os.walk(path):
        if path == root:
            for file in files:
                if file.find(filter) > -1:
                    file_list.append(os.path.join(root,file).replace("\\", "/"))
    return file_list

def hog_svm(img_path, show=False):
    im = cv2.imread(img_path)
    org_h, org_w = im.shape[:2]
    size = (512, int(512 * org_h / org_w))
    im = cv2.resize(im, size)
    #hog = cv2.HOGDescriptor()
    #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #hog = cv2.HOGDescriptor((32,64), (8,8), (4,4), (4,4), 9)
    #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #hog = cv2.HOGDescriptor((48,96), (16,16), (8,8), (8,8), 9)
    #hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())
    hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    locations, r = hog.detectMultiScale(im, winStride=(8, 8), padding=(32, 32), scale=1.05, hitThreshold=0, finalThreshold=1)
    for (x, y, w, h) in locations:
        cv2.rectangle(im, (x, y),(x+w, y+h),(255,255,0), 3)
    if show:
        cv2.imshow("detect image",im)
        cv2.waitKey(0)
    return im
    
if __name__ == '__main__':
    argv = sys.argv
    argc = len(argv)
    if (argc != 2):
        print ('Usage: python %s dir_path' %argv[0])
        quit()
    image_dir = argv[1]
    image_paths = getFilePath(image_dir, '.png')
    
    result_dir = './result'
    if os.path.exists(result_dir) == False:
        os.makedirs(result_dir)
    
    for image_path in image_paths:
        im = hog_svm(image_path)
        out_file_name = result_dir + "/result_" + os.path.basename(image_path)
        cv2.imwrite(out_file_name,im)
        print("done :", image_path)