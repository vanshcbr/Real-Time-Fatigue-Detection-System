import streamlit as st
import cv2
import os
from keras.models import load_model # type: ignore
import numpy as np
from pygame import mixer
import time

st.title("*SENTINEL*")

# Initialize sound mixer and load alarm sound
mixer.init()
sound = mixer.Sound(r'alarm.wav')

# Load Haar cascades for face and eye detection
face = cv2.CascadeClassifier(r'haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier(r'haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier(r'haar cascade files\haarcascade_righteye_2splits.xml')

# Load the pre-trained model
model = load_model(r'eye_status_cnn_model.h5')

# Start video capture
cap = cv2.VideoCapture(2)  
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0
thicc = 2
lbl = ['Closed', 'Open']

# Function for processing and displaying frames in Streamlit
def process_frame():
    global score, thicc

    ret, frame = cap.read()
    if not ret:
        st.warning("Camera not detected!")
        return None

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    rpred_class, lpred_class = 1, 1  # Defaults to "Open"
    
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        r_eye = cv2.resize(r_eye, (64, 64))
        r_eye = r_eye / 255.0
        r_eye = np.expand_dims(r_eye, axis=0)

        rpred = model.predict(r_eye)
        rpred_class = 1 if rpred[0][0] > 0.5 else 0
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.resize(l_eye, (64, 64))
        l_eye = l_eye / 255.0
        l_eye = np.expand_dims(l_eye, axis=0)

        lpred = model.predict(l_eye)
        lpred_class = 1 if lpred[0][0] > 0.5 else 0
        break

    if rpred_class == 0 and lpred_class == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

    if score < 0:
        score = 0

    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 5:
        try:
            sound.play()
        except:
            pass
        if thicc < 7:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    return frame

run_detection = st.checkbox("Driving")

# Placeholder for displaying frames
frame_placeholder = st.empty()

while run_detection:
    frame = process_frame()
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", use_column_width=True)
    else:
        break

cap.release()
cv2.destroyAllWindows()
