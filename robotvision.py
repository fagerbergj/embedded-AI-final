# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from threading import Thread
import numpy as np
import time
import cv2
import os
import pickle
import argparse

# get model and label paths from user
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help = "path to input model")
ap.add_argument("-l", "--labels", required=True, help="path to onput label binarizer")
args = vars(ap.parse_args())

MODEL_PATH = args["model"]
LABEL_PATH = args["labels"]

# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)
lb = pickle.loads(open(LABEL_PATH, "rb").read())

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

width = int(vs.get(3))  # width of frame
height = int(vs.get(4)) # height of frame

# muls effect size of crosshair, bigger muls -> bigger crosshair
widthMul = 4
heightMul = 7
# percent of screen covered by crosshair
widthSlice = int(width/widthMul)
heightSlice = int(height/heightMul)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream
    f = vs.read()
    frame = f[1]

    # prepare the image to be classified by our deep learning network
    # only take image in crosshair
    image = frame[heightSlice:(heightMul-1) * heightSlice, widthSlice : (widthMul-1) * widthSlice , :]
    # rezise image to 96x96
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # color values for ui
    r = 0
    g = 0
    b = 0
    # classify the input image and initialize the label and
    # probability of the prediction
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx].decode("utf-8")
    
    if proba[idx] < .7:
        label = "Unidentified Object"
        b = 255
    elif label == "robot":
        r = 255
    else:
        g = 255

    # build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, proba[idx] * 100)
    frame = cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (b, g, r), 2)
    frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[b, g, r])

    # draw crosshair
    lineThickness = 8
    cv2.line(frame, (widthSlice, int(height/2)), ((widthMul - 1)* widthSlice, int(height/2)), (b,g,r), lineThickness)
    cv2.line(frame, (int(width/2), heightSlice), (int(width/2), (heightMul - 1) * heightSlice), (b,g,r), lineThickness)

    # show the output frame
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()
