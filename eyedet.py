import cv2
import numpy as np

eyeCascade = cv2.CascadeClassifier("Resources/frontalEyes35x16.xml")

vid = cv2.VideoCapture(0)
# 0 for system's webcam and 1 for external webcam

while True:
    ret, img = vid.read()
    eyes = eyeCascade.detectMultiScale(img, 1.1, 4)
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv2.imshow("output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break