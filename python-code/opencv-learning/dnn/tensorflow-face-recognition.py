import tensorflow as tf
import numpy as np
import cv2

cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')

while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
        	cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        	roi_gray = gray[y:y+h, x:x+w]
        	roi_color = frame[y:y+h, x:x+w]
		cv2.imwrite('face.jpg',roi_color)
		im='/home/rey/Github/face_recognition_tensorflow/face.jpg'

        	# Read in the image_data
		image_data =tf.gfile.FastGFile(im, 'rb').read()

		#load retrained label data
        	label_lines = [line.rstrip() for line
       	                in tf.gfile.GFile("/home/rey/Github/face_recognition_tensorflow/retrained_labels.txt")]

		#load retrained graph
        	with tf.gfile.FastGFile("/home/rey/Github/face_recognition_tensorflow/retrained_graph.pb", 'rb') as f:
            		graph_def = tf.GraphDef()
            		graph_def.ParseFromString(f.read())
            		_ = tf.import_graph_def(graph_def, name='')

        	with tf.Session() as sess:
           		# Feed the image_data as input to the graph and get first prediction
            		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            		predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
	        	# Sort to show labels of first prediction in order of confidence
        		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

		info=str(label_lines[top_k[0]]) + ', Confidence:' + str(round((predictions[0][top_k[0]]*100)-0.05,2)) + '%'
		cv2.putText(frame, info, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	#show video stream
	cv2.imshow('Face Recognition', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break

print ('Session Ended')
cap.release()
cv2.destroyAllWindows()