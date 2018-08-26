#coding:utf-8
'''
使用MobileNetSSD的Caffe模型识别单个图片
'''

import numpy as np
import cv2

CLASSES = [line.strip() for line in open('../resources/models/caffe/object_detection_classes_pascal_voc.txt')]
print('[INFO]', CLASSES)
default_confidence = 0.2
# Generate random bounding box colors for each class label
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load Caffe model from disk
prototxt = '../resources/models/caffe/MobileNetSSD_deploy.prototxt'
model = '../resources/models/caffe/MobileNetSSD_deploy.caffemodel'
print("[INFO] Loading model")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print('[Info] cannot open video file')
    exit(0)
while True:
    # Load Image
    ret,frame = cap.read()
    if not ret:
        print('cannot read frame from video')
        break
    (h, w) = frame.shape[:2]
    # MobileNet requires fixed dimensions for input image(s)
    # so we have to ensure that it is resized to 300x300 pixels.
    # set a scale factor to image because network the objects has differents size.
    # We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 300, 300)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > default_confidence:
            # Extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            class_id = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # Draw bounding box for the object
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_id], 2)

            # Draw label and confidence of prediction in frame
            label = "{}: {:.2f}%".format(CLASSES[class_id], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                            COLORS[class_id], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[class_id], 2)

    # Show fame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
# Clean-up
cv2.destroyAllWindows()
