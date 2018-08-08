# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from threading import Thread
import numpy as np
import time
import cv2
import os
import pickle

MODEL_PATH = "Robots1.model"
LABEL_PATH = "lb.pickle"

# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)
lb = pickle.loads(open(LABEL_PATH).read())

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    f = vs.read()
    frame = f[1]
    # prepare the image to be classified by our deep learning network
    image = cv2.resize(frame, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    r = 0
    g = 0
    b = 0
    # classify the input image and initialize the label and
    # probability of the prediction
    proba = model.predict(image)
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    if idx < 80:
        label = "Unidentified Object"
        b = 255
    elif label == "robot":
        r = 255
    else:
        g = 255

    # build the label and draw it on the frame
    label = "{}: {:.2f}%".format(label, proba * 100)
    frame = cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (b, g, r), 2)
    border = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[b, g, r])
    # show the output frame
    #cv2.imshow("Frame", frame)
    cv2.imshow("border", border)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()
