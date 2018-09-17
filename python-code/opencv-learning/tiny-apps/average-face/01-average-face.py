import numpy as np
import cv2
import math
import sys, glob
import os

help_message = '''
USAGE:   averageface.py pathname
EXAMPLE: averageface.py /tmp/faces/*.jpg
OUTPUT:  average.jpg
'''

def detect(img, cascade):
    # Possible optimizations for detection
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)

    #rects = cascade.detectMultiScale(img)
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50), flags = 0)
    if len(rects) == 0:
        return []

    return rects

def draw_eyepositions(img, eyepositions, color):
    for x, y in eyepositions:
        cv2.rectangle(img, (x, y), (x, y), color, 2)

def detect_eyepositions_from_image(img, cascade):
    rects = detect(img, cascade)
    #print rects
    if len(rects) != 2:
        return []
    eyepositions = rects[:,:2] + rects[:,2:] / 2

    # sort array to contain left eye first, then right
    if eyepositions[0,0] > eyepositions[1,0]:
        eyepositions = eyepositions[::-1]

    return eyepositions

def get_imageinfo_from_files(files, cascade):
    result = []
    for filename in files:
        if os.path.isdir(filename):
            continue
        print "Processing " + filename
        img = cv2.imread(filename)
        width = img.shape[1]
        height = img.shape[0]
        eyepositions = detect_eyepositions_from_image(img, cascade)

        if len(eyepositions) == 2:
            result.append((filename, width, height, eyepositions))
        else:
            print "Could not find two eyes for " + filename + "! ignoring it."
    return result

def mergeimages(imageinfo, eye_center, target_width, target_height, cascade):
    print "Merging " + str(len(imageinfo)) + " images to size " + str(target_width) + "," + str(target_height)
    average_image = np.zeros([target_height, target_width, 3])
    average_eye_x_distance = eye_center[1][0] - eye_center[0][0]
    #print "average_eye_x_dist " + str(average_eye_x_distance)
    ok_faces = 0
    for filename, width, height, eyeposition in imageinfo:
        eye_x_distance = eyeposition[1][0] - eyeposition[0][0]
        #print "orig eye_distance " + str(eye_x_distance)
        resize_factor = average_eye_x_distance / float(eye_x_distance)
        #resize_factor = 1

        img = cv2.imread(filename)
        #print str(img.shape[0]) + " " + str(img.shape[1])
        if (resize_factor != 1):
            #print "Resizing " + filename + " with factor " + str(resize_factor)
            #print "original eyeposition " + str(eyeposition)
            img = cv2.resize(img, (int(img.shape[1] * resize_factor), int(img.shape[0] * resize_factor)))
            new_eyeposition = eyeposition * resize_factor
            eyeposition = new_eyeposition.astype(int)
            eye_x_distance = eyeposition[1][0] - eyeposition[0][0]
        diff = eye_center - eyeposition
        x_diff = (diff[0][0] + diff[1][0]) / 2
        y_diff = (diff[0][1] + diff[1][1]) / 2

        if (x_diff >= 0):
            source_x_offset = 0
            target_x_offset = x_diff
        else:
            source_x_offset = abs(x_diff)
            target_x_offset = 0

        if (y_diff >= 0):
            source_y_offset = 0
            target_y_offset = y_diff
        else:
            source_y_offset = abs(y_diff)
            target_y_offset = 0

        max_width = min(target_width, img.shape[1])
        max_height = min(target_height, img.shape[0])
        print filename + " " + str(resize_factor)
        x_size_old = img.shape[1]
        y_size_old = img.shape[0]

        x_size = max_width - abs(x_diff)
        y_size = max_height - abs(y_diff)
        #print str(x_size_old) + " " +  str(y_size_old) + " " + x_size + " " + y_size

        temp_image = np.empty([target_height, target_width, 3])
        temp_image.fill(255)
        #print str(target_y_offset) + " " + str(y_size) + " " + str(target_x_offset) + " "+ str(x_size)
        #print str(source_y_offset) + " " + str(source_x_offset)
        temp_image[target_y_offset:target_y_offset + y_size, target_x_offset:target_x_offset + x_size] = img[source_y_offset:source_y_offset + y_size, source_x_offset:source_x_offset + x_size]
        average_image += temp_image
        ok_faces += 1
    average_image /= ok_faces
    average_image = average_image.astype(float)
    print "Generated average image using " + str(ok_faces) + " images"
    return average_image

def calculate_eye_center(imageinfo):
    eye_center = [[0,0], [0,0]]
    for e in imageinfo:
        eye_center += e[3]
    eye_center /= len(imageinfo)
    return eye_center

def calculate_average_size(imageinfo):
    average_width = 0
    average_height = 0
    for e in imageinfo:
        average_width += e[1]
        average_height += e[2]

    average_width /= len(imageinfo)
    average_height /= len(imageinfo)
    return average_width, average_height

if len(sys.argv) != 2:
    print help_message
    sys.exit(1)

pathname = sys.argv[1]
files = glob.glob(pathname)

classifier_filename = "haarcascade_eye_tree_eyeglasses.xml"

cascade = cv2.CascadeClassifier(classifier_filename)

imageinfo = get_imageinfo_from_files(files, cascade)
if len(imageinfo) < 2:
    print "Sorry, could not find enough images for processing. At least two needed."
    sys.exit(1)

#print imageinfo
eye_center = calculate_eye_center(imageinfo)
#print "eye_center" + str(eye_center)

average_width, average_height = calculate_average_size(imageinfo)
average_image = mergeimages(imageinfo, eye_center, average_width, average_height, cascade)
cv2.imwrite("average.jpg", average_image)