import cv2
from pyimagesearch.utils import Conf
from datetime import datetime
import face_recognition
import numpy as np
import argparse
import imutils
import pickle
import signal
import time
import cv2
import sys
import os

'''
# If you want to have fun with LEDs and switch it on if your face was detected:

import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
my_pin = 24
GPIO.setup(my_pin,GPIO.OUT)

def led_on(pin):
    GPIO.output(pin,GPIO.HIGH)

def led_off(pin):
    GPIO.output(pin,GPIO.LOW)
'''

# Camera capture -- we have only one Pi camera, hence "0" as the ID
cam = cv2.VideoCapture(0)

# function to handle keyboard interrupt
def signal_handler(sig, frame):
    print("[INFO] You pressed `ctrl + c`! Closing face recognition")
    sys.exit(0)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
    help="Path to the input configuration file")
args = vars(ap.parse_args())

# load the configuration file and initialize the Twilio notifier
conf = Conf(args["conf"])

# load the actual face recognition model, label encoder, and face detector
recognizer = pickle.loads(open(conf["recognizer_path"], "rb").read())
le = pickle.loads(open(conf["le_path"], "rb").read())
detector = cv2.CascadeClassifier(conf["cascade_path"])

print("Available class labels:")
for c in le.classes_:
    print("\t{}".format(c))

while True:

    retval, img = cam.read()
    res_scale = 0.3             # rescale the input image if it's too large
    frame = cv2.resize(img, (0, 0), fx=res_scale, fy=res_scale)

    # We got the frame, let's process it and show the results:

    # *** Task 1. Convert the frame to grayscale image and RGB (from BGR) using cv2.cvtColor(...)
    # 
    gray = cv2.cvtColor(frame, cv2.BGR2GRAY)    # this will be used by Viola-Jones detector
    rgb = cv2.cvtColor(frame, cv2.BGR2RGB)     # this will be used by ResNet-based face encoder


    # *** Task 2. Run detectMultiScale(...) method from the "detector" to get a vector of rectangles, 
    # possibly containing faces:
    # 
    rectangles = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    # 
    # arguments of detectMultiScale(...):
    # scaleFactor: how much the image size is reduced at each image scale. Used to create the VJ scale pyramid (1.05 -- 1.4)
    # minNeighbors: how many neighbors each candidate rectangle should have to retain it (3 -- 6)
    # minSize = (N, M): minimum possible face size (depends on your resolution, but (30,30) should be fine)

    # Let's get boxes for all detected faces:
    boxes = [ ( y, x + w, y + h, x ) for x, y, w, h in rectangles ]
    print("boxes: {}".format(np.shape(boxes)))

    # *** Task 3. For each box in the "rectangles" do the face recognition:

    if len(boxes) > 0:

        # compute the facial embedding for the face, as you did for the dataset of faces
        encodings = face_recognition.face_encodings(rgb, boxes)

        # compute the facial embeddings for all boxes (= detected faces)
        preds = recognizer.predict_proba(encodings)
        DEBUG: print("predictions: {}".format(np.shape(preds)))

        # So what have so far:
        # -- boxes: a (N,4) matrix of N boxes for N detected faces
        # -- preds: a (N,M) matrix of M probabilities for each of N detected faces
        # -- le.classes_[index] will give us the name of the class, given the index
        # 
        # *** Task: 
        # -- iterate over all boxes and preds (tip: use "zip")
        # -- find the class name (= person's name, as in the "database" folder) 
        #    by searching for the class index with the highest probability 
        # -- display the result on screen (box + the person's name)
        #    tip: use cv2.rectangle(...) and cv2.putText(...) functions

    cv2.imshow("Face detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()