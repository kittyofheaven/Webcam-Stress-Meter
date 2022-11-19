import numpy as np
import cv2
import sys

from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.utils import img_to_array , load_img
from keras.models import  load_model

# Model Load
model = load_model("model.h5")
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# webcam choose
webcam = None
if len(sys.argv) == 2:
    webcam = cv2.VideoCapture(sys.argv[1])
else:
    webcam = cv2.VideoCapture(0)

# window scale stuff
realWidth = 320
realHeight = 240
videoWidth = 160
videoHeight = 120
videoChannels = 3
videoFrameRate = 15
webcam.set(3, realWidth)
webcam.set(4, realHeight)

# color
levels = 3
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 150
bufferIndex = 0

# display
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255,255,255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

while True:
    ret, frame = webcam.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if ret == False:
        break
    faces_detected = face_haar_cascade.detectMultiScale(frame, 1.32, 5)
    cv2.putText(frame, "Place your head in the guide box for better result", (0,10), font, 0.3, fontColor, 1)

    # Guide box
    cv2.rectangle(frame, (videoWidth//2 , videoHeight//2), (realWidth-videoWidth//2, realHeight-videoHeight//2), (0,0,0), 1)

    for (x, y, w, h) in faces_detected:
        if len(sys.argv) != 2:
            originalFrame = frame.copy()
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        emotionsTextLocation = (int(x), int(y-5))
        cv2.putText(frame, "Emotions: %s" % predicted_emotion, emotionsTextLocation , font, fontScale, fontColor, lineType)
    cv2.imshow('Stress Monitor', frame)


