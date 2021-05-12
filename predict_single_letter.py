# organize imports
import cv2
import imutils
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import numpy as np
import pickle

from pathlib import Path
from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split

from skimage.io import imread
from skimage.transform import resize
import skimage

with open('ISL_SVM_SAVEFILE.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model Loaded")

dimension = (64,64)

def predict_single_image(image_name):
    alpha = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'SPACE', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'}
    img = skimage.io.imread(image_name, as_gray=True)
    img = resize(img, dimension, anti_aliasing=True, mode='reflect')
    img = img.flatten()
    img = img.reshape(1,-1)
    class_probabilities = list(model.predict_proba(img)[0])
    probability = max(class_probabilities)
    index = class_probabilities.index(probability)
    class_name = alpha[index]
    return (index, class_name, probability)



# global variables
bg = None
captureMode = False
text_to_display = "Start with S"
confidence_threshold = 70

#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    ( cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#-----------------
# MAIN FUNCTION
#-----------------
if __name__ == "__main__":
    messageDisplayed = False
    
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 80, 320, 280, 520

    # initialize num of frames
    num_frames = 0

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        #frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            if not messageDisplayed:
                print("Place Hand in the green box")
                print("To start prediction press S")
                messageDisplayed = True
            
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

                if(captureMode):
                    cv2.imwrite("WEBCAM.jpg",thresholded)
                    class_no, class_name, confidence = predict_single_image("WEBCAM.jpg")
                    text_to_display = "Class: " + class_name + "    Confidence: " + "{:.2f}".format(confidence*100) + "%"
                    if(int(confidence*100)>confidence_threshold):
                        print(text_to_display)
                
        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

        # increment the number of frames
        num_frames += 1

        clone = cv2.putText(clone, text_to_display, (50, 50), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (255, 0, 0) , 2, cv2.LINE_AA) 

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("s"):
            captureMode = not captureMode
            print("Capturing", captureMode)
            

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

# free up memory
camera.release()
cv2.destroyAllWindows()


