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

# args
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
    help="path to input model")
ap.add_argument("-l", "--label", required=True,
    help="path to input label binarizer")
ap.add_argument("-c", "--confidence", type=float, default=.7,
    help="threashold for confidence in labeling")
ap.add_argument("-t", "--crosshair_thickness", required=False, type=int, default=4,
    help="thickness of crosshair")
ap.add_argument("-wc", "--crosshair_width_frac", required=False, type=int, default=3,
    help="fraction not covered on left/right of crosshair")
ap.add_argument("-hc", "--crosshair_height_frac", required=False, type=int, default=4,
    help="fraction not covered on top/bottom of crosshair")
args = vars(ap.parse_args())

# path to needed files
MODEL_PATH = args["model"]
LABEL_PATH = args["label"]

# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)
lb = pickle.loads(open(LABEL_PATH, "rb").read())

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

#width of screen
width = int(vs.get(3))  # float
height = int(vs.get(4)) # float

#section off screen to only get crosshair
widthMul = args["crosshair_width_frac"]
heightMul = args["crosshair_height_frac"]
widthSlice = int(width/widthMul)
heightSlice = int(height/heightMul)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    f = vs.read()
    frame = f[1]
    # prepare the image to be classified by our deep learning network
    image = frame[heightSlice:(heightMul-1) * heightSlice, widthSlice : (widthMul-1) * widthSlice , :]
    #print(image.shape)
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    r = 0
    g = 0
    b = 0
    # classify the input image and initialize the label and
    # probability of the prediction
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx].decode("utf-8")
    conf = args["confidence"]

    if proba[idx] < conf:
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

    lineThickness = args["crosshair_thickness"]
    cv2.line(frame, (widthSlice, int(height/2)), ((widthMul - 1)* widthSlice, int(height/2)), (b,g,r), lineThickness)
    cv2.line(frame, (int(width/2), heightSlice), (int(width/2), (heightMul - 1) * heightSlice), (b,g,r), lineThickness)

    # show the output frame
    #cv2.imshow("Frame", frame)
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()

