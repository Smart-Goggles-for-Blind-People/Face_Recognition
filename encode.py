import cv2
import face_recognition
import pickle
import os

folderPath= 'images'
pathList = os.listdir(folderPath)
#print(pathList)
imgList = []
familyIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))

    familyIds.append(os.path.splitext(path)[0])
   # print(os.path.splitext(path)[0])
print(familyIds)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList
print("Encoding...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, familyIds]
print(encodeListKnown)
print("Encoded")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds ,file)
file.close()