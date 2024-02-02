import time
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
from gtts import gTTS
import os
import pygame
from deepface import DeepFace

last_recognition_time = {}
delay_time = 60
absolute_file_path = "/Users/omkar/PycharmProjects/object detection/venv/EncodeFile.p"


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


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load encoding file
print("Loading Encode File...")
file = open(absolute_file_path, 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, familyIds = encodeListKnownWithIds
print("Encoded file loaded..")
faceNet = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
genderNet = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

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

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        bbox = x1, y1, x2 - x1, y2 - y1

        img = cvzone.cornerRect(img, bbox, rt=0)

        if matches[matchIndex]:
            # Known face
            name_to_speak = familyIds[matchIndex]

            # Check if 60 sec pass then recognize
            current_time = time.time()
            if name_to_speak not in last_recognition_time or current_time - last_recognition_time[
                name_to_speak] >= delay_time:
                print(name_to_speak)

                # Emotion recognition
                face_img = img[int(y1):int(y2), int(x1):int(x2)]
                emotion_result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                emotion = emotion_result[0]['dominant_emotion']
                print(f"{name_to_speak}'s emotion: {emotion}")

                speak(f"{name_to_speak}'s emotion is {emotion}")
                last_recognition_time[name_to_speak] = current_time

        else:
            # Unknown face
            name_to_speak = "Unknown Person"

            # Check if 60 sec pass then recognize
            current_time = time.time()
            if name_to_speak not in last_recognition_time or current_time - last_recognition_time[
                name_to_speak] >= delay_time:
                print(name_to_speak)

                # Emotion recognition for unknown person
                face_img = img[int(y1):int(y2), int(x1):int(x2)]
                emotion_result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
                emotion = emotion_result[0]['dominant_emotion']
                print(f"{name_to_speak}'s emotion: {emotion}")


                last_recognition_time[name_to_speak] = current_time

                face_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), [104, 117, 123], swapRB=False)
                genderNet.setInput(face_blob)
                gender_preds = genderNet.forward()
                gender = "Male" if gender_preds[0].argmax() == 0 else "Female"
                print(f"{name_to_speak}'s gender: {gender}")

                speak(f"I think {name_to_speak}'s emotion is {emotion} and gender is {gender}")
                last_recognition_time[name_to_speak] = current_time

    # Move these lines outside the for loop
    cv2.imshow("Project For Blind People", img)
    cv2.waitKey(1)
