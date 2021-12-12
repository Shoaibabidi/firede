import cv2
import numpy as np
from playsound import playsound
import streamlit as st

fire_cascade = cv2.CascadeClassifier('fire.xml')
cap = cv2.VideoCapture(0)
st.title('Detector')

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(frame, 1.2, 5)

    for (x,y,w,h) in fire:
        cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        print("fire is detected")
        playsound("audio.mp3")
        st.text('Fire Detected!')

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
frameST = st.empty()
param=st.sidebar.slider('chose your value')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Done processing !!!")
        cv2.waitKey(3000)
        cap.release()
        break

    frameST.image(frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()