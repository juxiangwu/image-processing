import facemesh as fm
import cv2

def main():

    #Instantiate the class
    fullSet = "C:/Users/fleis/Desktop/portraits"
    testSet = "C:/Users/fleis/Desktop/test"
    faceReader = fm.FaceMesh("shape_predictor_68_face_landmarks.dat")
    outputImgs = faceReader.align(fullSet)

    #Save all the alined images to output folder
    counter = 0
    for img in outputImgs:
        counter+= 1
        cv2.imwrite("output/" + str(counter) + ".png", img.astype('uint8'))

    #Save one average image
    av = faceReader.getAvarage()
    cv2.imwrite("avrage.png", av.astype('uint8'))
if __name__ == '__main__':
    main()