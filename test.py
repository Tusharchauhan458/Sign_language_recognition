import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            Prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(Prediction, index)
        else:
            k = imgSize / w
            wCal = math.ceil(k * w)
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            Prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_+
                    
                    COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

