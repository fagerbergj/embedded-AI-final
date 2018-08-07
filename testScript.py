# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import imutils
import time
import cv2
import os

MODEL_PATH = "Robots1.model"

# load the model
print("[INFO] loading model...")
model = load_model(MODEL_PATH)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
        # grab the frame from the threaded video stream and resize it
    	# to have a maximum width of 400 pixels
    	frame = vs.read()
        frame = imutils.resize(frame, width=400)
        # prepare the image to be classified by our deep learning network
    	image = cv2.resize(frame, (28, 28))
    	image = image.astype("float") / 255.0
    	image = img_to_array(image)
    	image = np.expand_dims(image, axis=0)
        # classify the input image and initialize the label and
    	# probability of the prediction
    	(robot, human) = model.predict(image)[0]
    	label = "Robot"
    	proba = robot
        # check to see if santa was detected using our convolutional
    	# neural network
    	if human > robot:
            # update the label and prediction probability
            label = "Human"
            proba = human
        if human < 50 and robot < 50:
            label = "Unable to determine"
            proba = 0
        # build the label and draw it on the frame
    	label = "{}: {:.2f}%".format(label, proba * 100)
    	frame = cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    	# show the output frame
    	cv2.imshow("Frame", frame)
    	key = cv2.waitKey(1) & 0xFF
    	# if the `q` key was pressed, break from the loop
    	if key == ord("q"):
            break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
