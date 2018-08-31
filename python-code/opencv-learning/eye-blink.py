#coding:utf-8
import cv2


# Initialize cascades
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

# Primitive detectors
def _get_faces(image):
	return face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

def _get_eyes(image):
	return eye_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)


# Actual detector
class Eye:

	def __init__(self, bounds=[0] * 4):
		self.bounds = bounds

	def is_closed(self):
		return self.bounds[0] == self.bounds[2]

class Face:

	def __init__(self, bounds=[0] * 4, left_eye=Eye(), right_eye=Eye()):
		self.bounds = bounds
		self.left_eye = left_eye
		self.right_eye = right_eye


def get_faces(image):
	'''Detects faces (and eyes) in an image
	# Arguments:
	image: Image in which faces are to be detected
	# Returns: List of Face objects.
	# Example:
		face = get_faces(image)[0]
		if face.right_eye.is_closed():
			print("Either this guy is a pirate, or his right eye closed.")
	'''
	faces = []
	face_rects = _get_faces(image)
	for face_rect in face_rects:
		roi = image[face_rect[1]:face_rect[1] + face_rect[3], face_rect[0] : face_rect[0] + face_rect[2]]
		eye_rects = _get_eyes(roi)
		for i in range(len(eye_rects)):
			eye_rects[i] = [face_rect[0] + eye_rects[i][0] , face_rect[1] + eye_rects[i][1] , face_rect[0] + eye_rects[i][0] + eye_rects[i][2], face_rect[1] + eye_rects[i][1] + eye_rects[i][3]]
		if len(eye_rects) == 0:
			faces += [Face([face_rect[0], face_rect[1], face_rect[0] + face_rect[2], face_rect[1] + face_rect[3]])]
		elif len(eye_rects) == 1:
			if eye_rects[0][0] + eye_rects[0][2] < 2 * face_rect[0] + face_rect[2]:
				faces += [Face([face_rect[0], face_rect[1], face_rect[0] + face_rect[2], face_rect[1] + face_rect[3]], right_eye=Eye(eye_rects[0]))]
			else:
				faces += [Face([face_rect[0], face_rect[1], face_rect[0] + face_rect[2], face_rect[1] + face_rect[3]], left_eye=Eye(eye_rects[0]))]
		else:
			if eye_rects[0][0] < eye_rects[1][0]:
				eye_rects = [eye_rects[1], eye_rects[0]]
			faces += [Face([face_rect[0], face_rect[1], face_rect[0] + face_rect[2], face_rect[1] + face_rect[3]], left_eye=Eye(eye_rects[0]), right_eye=Eye(eye_rects[1]))]
	return faces


# Initialize camera
camera = cv2.VideoCapture(0)


# Main Loop
while(True):
	# Read camera input
	flag, image = camera.read()
	# Resize image to width 300
	image = cv2.resize(image, (300, int(image.shape[0] * 300. / image.shape[1])), interpolation=cv2.INTER_AREA)
	# Convert to gray scale image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = get_faces(gray)
	# Define bounding box colors
	left_eye_box_color = (0, 255, 0) # Green
	right_eye_box_color = (255, 0, 0) # Blue
	face_box_color = (0, 0, 255) # Red
	# Draw boxes
	for face in faces:
		eyes = []
		cv2.rectangle(image, (face.bounds[0],face.bounds[1]), (face.bounds[2],face.bounds[3]), face_box_color, 2)
		if not face.left_eye.is_closed():
			eyes += ['Left']
			cv2.rectangle(image, (face.left_eye.bounds[0],face.left_eye.bounds[1]), (face.left_eye.bounds[2],face.left_eye.bounds[3]), left_eye_box_color, 2)
		if not face.right_eye.is_closed():
			eyes += ['Right']
			cv2.rectangle(image, (face.right_eye.bounds[0],face.right_eye.bounds[1]), (face.right_eye.bounds[2],face.right_eye.bounds[3]), right_eye_box_color, 2)
	# Print text
	text_color = (0, 0, 255)
	if len(faces) == 0:
		text = "No faces detected!"
	elif len(faces) == 1:
		if len(eyes) == 2:
			text = "Both eyes open"
		elif len(eyes) == 0:
			text = "Face detected"
		else:
			if eyes[0] == 'Left':
				text_color = (0, 255, 0)
				text = 'Left eye open'
			else:
				text_color = (255, 0, 0)
				text = 'Right eye open'
	else:
		text = 'Multiple faces detected'
	cv2.putText(image,text, (0, int(0.9 * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color)
	cv2.putText(image,"Press 'x' to quit", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
	# Display final image
	cv2.imshow("Blink Demo", image)
	# Wait for exit key
	if cv2.waitKey(1) & 0xFF == ord("x"):
		break

# Release resources
camera.release()
cv2.destroyAllWindows()