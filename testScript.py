# import necessary packages 
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2
import os
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
ap.add_argument("-l", "--labelbin", required = True, help="path to label binarizer")
ap.add_argument("-i", "--image", required=True, help = "path to input image")
args = vars(ap.parse_args())

#load image
image = cv2.imread(args["image"])
output = image.copy()

#preprocess the image
image = cv2.resize(image, (96, 96))
image = image.astype("float")/255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

#load the trained model
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

while True:
    #classify the input image
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]

    # mark prediction as correct if the input filename contains the predicted label text 
    #filename = args["image"][args["image"].rfind(os.path.sep)+1:]
    #correct = "correct" if filename.rfind(label) != -1 else "incorrect"

    frame = vs.read()
    # fuck his imutils shit
    frame = imutils.resize(frame, width=400)

    #build the label and draw the label on the image
    label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
    output = imutils.resize(output, width=400)
    cv2.putText(output, label, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)

    #print("[INFO] {}".format(label))
    cv2.imshow("Output", output)
    #cv2.waitKey(0)