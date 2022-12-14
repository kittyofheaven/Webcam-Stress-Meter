import math
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


#####PARAMETERS START##### (ONLY CHANGE THIS!)

# webcam
webcam = None
if len(sys.argv) == 2:
    webcam = cv2.VideoCapture(sys.argv[1])
else:
    webcam = cv2.VideoCapture(0)
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

bpm_low_threshold = 100
bpm_high_threshold = 0

#####PARAMETERS END#####

#Gaussian Pyramid
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid
def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame

firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# BPM Variables
bpmCalculationFrequency = 15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

# stress variable 0-10
stress = 0
bpm_list = []
def stress_count(bpm, emotions_index):
    emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad')
    global stress
    global bpm_list

    bpm_list.append(bpm)
    mean = np.mean(bpm_list)
    mean_diff = math.floor(bpm-mean)
    # diff = bpm_high_threshold + bpm_low_threshold
    # baseline = diff / 2 + 1
    # baseline_diff = bpm - baseline

    if emotions_index == 4 : #neutral        
        stress = mean_diff
        if mean_diff > 10: 
            stress = 10
        if mean_diff < 0 :
            stress = 0 

    elif emotions_index == 0 or emotions_index == 1 or emotions_index == 2 : #angry, disgust, fear
        stress = 2 * mean_diff
        if mean_diff > 10: 
            stress = 10
        if mean_diff < 0 :
            stress = 0 

    elif emotions_index == 3 : #happy
        stress = 0.5 * mean_diff
        if mean_diff > 10: 
            stress = 10
        if mean_diff < 0 :
            stress = 0 
    return stress


#####CORE START#####
i = 0
while (True):
    ret, frame = webcam.read()
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if ret == False:
        break
    faces_detected = face_haar_cascade.detectMultiScale(frame, 1.32, 5)
    cv2.putText(frame, "Place your head in the center of the box for better result", (0,10), font, 0.3, fontColor, 1)

    for (x, y, w, h) in faces_detected:
        if len(sys.argv) != 2:
            originalFrame = frame.copy()

        # detect face and scan it
        cv2.rectangle(frame, (x, y), (x + w, y + h), boxColor, thickness=boxWeight)
        cropped = frame[y:y + w, x:x + h]
        # cv2.imshow("Monitor", cropped)
        detectionFrame = frame[videoHeight//2:realHeight-videoHeight//2, videoWidth//2:realWidth-videoWidth//2, :]
        
        # Construct Gaussian Pyramid
        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
        fourierTransform = np.fft.fft(videoGauss, axis=0)
        fourierTransform[mask == False] = 0

        # Pulse detection
        if bufferIndex % bpmCalculationFrequency == 0:
            i = i + 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha

        # emotions detections frame
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'sad')
        predicted_emotion = emotions[max_index]

        # Reconstruct Resulting Frame
        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
        outputFrame = detectionFrame 
        outputFrame = cv2.convertScaleAbs(outputFrame)
        bufferIndex = (bufferIndex + 1) % bufferSize
        frame[videoHeight//2:realHeight-videoHeight//2, videoWidth//2:realWidth-videoWidth//2, :] = outputFrame

        bpmTextLocation = (int(x), int(y-35))
        emotionsTextLocation = (int(x), int(y-20))
        stressTextLocation = (int(x), int(y-5))
        
        if i > bpmBufferSize:
            cv2.putText(frame, "BPM     : %d" % bpmBuffer.mean(), bpmTextLocation , font, fontScale, fontColor, lineType)
            cv2.putText(frame, "Emotions: %s" % predicted_emotion, emotionsTextLocation , font, fontScale, fontColor, lineType)
    
            bpm = int(bpmBuffer.mean())

            if bpm < bpm_low_threshold :
                bpm_low_threshold = bpm
            if bpm > bpm_high_threshold :
                bpm_high_threshold = bpm
            
            stress = stress_count(bpmBuffer.mean(), max_index)

            cv2.putText(frame, "Low :%d" % bpm_low_threshold, (0,25), font, 0.4, fontColor, 1)
            cv2.putText(frame, "High :%d" % bpm_high_threshold, (0,40), font, 0.4, fontColor, 1)
            if stress < 5:
                cv2.putText(frame, "Stress :%d" % stress, stressTextLocation, font, fontScale, fontColor, lineType)
            elif stress < 7 :
                cv2.putText(frame, "Stress :%d" % stress, stressTextLocation, font, fontScale, (0,255,255), lineType)
            else : 
                cv2.putText(frame, "Stress :%d" % stress, stressTextLocation, font, fontScale, (0,0,255), lineType)
                cv2.putText(frame, "You should take a rest!", (0,55), font, fontScale, (0,0,255), lineType)
        else:
            cv2.putText(frame, "Calculating ", stressTextLocation, font, fontScale, fontColor, lineType)

    if len(sys.argv) != 2:
        
        cv2.imshow("Webcam Stress Meter", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
#####CORE END#####
webcam.release()
cv2.destroyAllWindows()
