import cv2 as cv
import time
import numpy as np

classifier = cv.CascadeClassifier('/Users/ashwinjain/Development/Personal/gate_security/project/data/car_classifier.xml')


video = cv.VideoCapture('/Users/ashwinjain/Development/Personal/gate_security/project/data/cars.mp4')


while video.isOpened():
    time.sleep(.05)
    
    ret, frame = video.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cars = classifier.detectMultiScale(gray, 1.4, 2)

    for (x, y, w, h) in cars:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv.imshow('Cars', frame)
    
    if cv.waitKey(1) == 1:
        break

video.release()
cv.destroyAllWindows()