import cv2
import dlib
import numpy

#INITIALIZE DLIB FACE DETECTOR AND CREATE FACE LANDMARKS PREDICTOR
frontal_face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(
    "../resources/models/dlib/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

#GET INTERSECTION BETWEEN TWO LINES WITH COORDINATES ([x1,x2],[x2,y2])([x3,y3][x4,y4])
def get_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    denom = float(y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    #if denom == 0 there is no slope, but in our case there will always be
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    #ub = float((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    return (int(x), int(y))


def scale_faceangle(points, scale=1, offset=(0, 0)):
    mid = numpy.mean(points, axis=0)
    pts = []
    for i in range(len(points)):
        pts.append(
            tuple(
                numpy.array(
                    (numpy.subtract(
                        numpy.add(numpy.subtract(points[i], mid) * scale, mid),
                        offset)),
                    dtype=int)))
    return pts
    

def shape_to_np(shape):
    # initialize the list of (x, y)-coordinates
    coords = numpy.zeros((shape.num_parts, 2), dtype=int)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

while(True):
    # Capture frame-by-frame
    edgeTR, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #DETECT FACES IN THE GRAYSCALE FRAME
    faces = frontal_face_detector(gray, 0)

    for face in faces:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = shape_predictor(gray, face)
        #PREDICT FACE LANDMARKS AND CONVERT THEM TO NUMPY ARRAY COORDINATES
        shape = shape_to_np(shape_predictor(gray, face))

        for i in range(1, 7):
            #LEFTEST EYE POINT
            eyeL = tuple(shape[36])
            #RIGHTEST EYE POINT
            eyeR = tuple(shape[45])
            #MIDDLE EYE POINT
            eyeM = tuple(shape[27])
            #NOSE TOP POINT
            noseT = tuple(numpy.mean((numpy.mean((shape[21], shape[22]), axis=0), eyeM), axis=0))

            #NOSE BOTTOM POINT
            noseB = tuple(shape[33])
            #UPPER LIP BOTTOM MID POINT
            lipU = tuple(shape[62])
            #LOWER LIP TOP MID POINT
            lipL = tuple(shape[66])

            #CHIN BOTTOM POINT
            chinB = tuple(shape[8])
            tmp = numpy.subtract(numpy.mean((shape[6], shape[9]), axis=0),chinB)
            #CHIN LEFT POINT; CALCULATING MORE PRECISE ONE
            chinL = tuple(numpy.subtract(numpy.mean((shape[6], shape[7]), axis=0),tmp))
            #CHIN RIGHT POINT; CALCULATING MORE PRECISE ONE
            chinR = tuple(numpy.subtract(numpy.mean((shape[9], shape[10]), axis=0),tmp))

            #THE DIFFERENCE (eyeM - chinB) EQUALS 2/3 OF THE FACE
            tmp = numpy.subtract(eyeM, chinB)
            #GET 1/3 OF THE FACE
            tmp = tuple([int(x / 2) for x in tmp])

            #CALCULATING THE EDGES FOR THE BOX WE ARE GOING TO DRAW
            #EDGE POINT TOP LEFT, LEFT EYEBROW + 1/3 OF THE FACE SO WE GET THE FOREHEAD LINE
            edgeTL = tuple(numpy.add(shape[19], tmp))
            #EDGE POINT TOP RIGHT, RIGHT EYEBROW + 1/3 OF THE FACE SO WE GET THE FOREHEAD LINE
            edgeTR = tuple(numpy.add(shape[24], tmp))

            #MOVE THE TOP LEFT EDGE LEFT IN LINE WITH THE CHIN AND LEFT EYE - ESTIMATING FOREHEAD WIDTH
            edgeTL = get_intersection(edgeTL[0], edgeTL[1], edgeTR[0], edgeTR[1], eyeL[0],
                                   eyeL[1], chinB[0], chinB[1])
                                   
            #MOVE THE TOP RIGHT EDGE RIGHT IN LINE WITH THE CHIN AND RIGHT EYE - ESTIMATING FOREHEAD WIDTH
            edgeTR = get_intersection(edgeTR[0], edgeTR[1], edgeTL[0], edgeTL[1], eyeR[0],
                                   eyeR[1], chinB[0], chinB[1])

            tmp = numpy.subtract(eyeM, chinB)

            #EDGE POINT BOTTOM LEFT, CALCULATE HORIZONTAL POSITION
            edgeBL = tuple(numpy.subtract(edgeTL, tmp))
            #EDGE POINT BOTTOM RIGHT, CALCULATE HORIZONTAL POSITION
            edgeBR = tuple(numpy.subtract(edgeTR, tmp))

            #EDGE POINT BOTTOM LEFT, CALCULATE VERTICAL POSITION - IN LINE WITH CHIN SLOPE
            edgeBL = get_intersection(edgeTL[0], edgeTL[1], edgeBL[0], edgeBL[1], chinL[0], chinL[1],
                                   chinR[0], chinR[1])
            #EDGE POINT BOTTOM RIGHT, CALCULATE VERTICAL POSITION - IN LINE WITH CHIN SLOPE
            edgeBR = get_intersection(edgeTR[0], edgeTR[1], edgeBR[0], edgeBR[1], chinR[0], chinR[1],
                                   chinL[0], chinL[1])

            #CALCULATE HEAD MOVEMENT OFFSET FROM THE CENTER, lipU - lipL IS THE DISTANCE FROM BOTH LIPS (IN CASE MOUTH IS OPEN)
            offset = (float(noseT[0] - 2 * noseB[0] + chinB[0] + lipU[0]-lipL[0]) * 4,
                      float(noseT[1] - 2 * noseB[1] + chinB[1] + lipU[1]-lipL[1]) * 4)

            #BACKGROUND RECTANGLE
            recB = (edgeTL, edgeTR, edgeBR, edgeBL)
            
            #FOREBACKGROUND RECTANGLE
            recF = (scale_faceangle((recB), 1.5, offset))

            #DRAW FACIAL LANDMARK COORDINATES
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (255, 0, 255), 5)

            #DRAW BACKGROUND RECTANGLE
            cv2.polylines(frame, numpy.array([recB], numpy.int32), True,
                          (255, 0, 0), 5)

            #DRAW FACE BOX EDGE LINES
            for i in range(4):
                cv2.line(frame, recB[i], recF[i], (255, 255, 0), 5)

            #DRAW NOSE DIRECTION LINE
            cv2.line(
                frame, tuple(shape[30]),
                tuple(
                    numpy.array(
                        (numpy.subtract(shape[30], offset)), dtype=int)),
                (0, 255, 255), 5)

            #DRAW FOREGROUNDBACKGROUND RECTANGLE
            cv2.polylines(frame, numpy.array([recF], numpy.int32), True,
                          (0, 255, 0), 5)

    cv2.imshow('Frame', frame)

    #PRESS ESCAPE TO EXIT
    if cv2.waitKey(1) == 27:
        break

#RELEASE THE CAP
cap.release()
cv2.destroyAllWindows()