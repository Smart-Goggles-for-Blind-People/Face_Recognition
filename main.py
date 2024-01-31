import time
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
from gtts import gTTS
import os
import pygame


last_recognition_time = {}
delay_time = 60

def speak(text):
    tts_text = text + " is in front of you.."
    tts = gTTS(text=tts_text, lang='en')
    tts.save('temp.mp3')
    pygame.mixer.init()
    pygame.mixer.music.load('temp.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue
    pygame.mixer.quit()
    os.remove('temp.mp3')

cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Load encoding file
print("Loading Encode File...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, familyIds = encodeListKnownWithIds
print("Encoded file loaded..")

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name_to_speak = familyIds[matchIndex]

            # Check if 60 sec pass then recognise
            current_time = time.time()
            if name_to_speak not in last_recognition_time or current_time - last_recognition_time[name_to_speak] >= delay_time:
                print(name_to_speak)
                speak(name_to_speak)
                last_recognition_time[name_to_speak] = current_time

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            bbox = x1, y1, x2 - x1, y2 - y1

            img = cvzone.cornerRect(img, bbox, rt=0)

    cv2.imshow("Project For Blind People", img)
    cv2.waitKey(1)
