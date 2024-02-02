import cv2
import face_recognition
import pickle
import os



###################################################################################################
################## "firebase Intialze" #####################################


#import firebase_admin

#from firebase_admin import db
#from firebase_admin import credentials, storage


#cred = credentials.Certificate("/Users/omkar/Downloads/credentials.json")
#firebase_admin.initialize_app(cred, options={
    #'storageBucket': "smart-goggle-9e64d.appspot.com"
#})



##########################################################################################################
################### Uploading Image #############################################################


#bucket = storage.bucket()

#local_image_path = '/Users/omkar/Downloads/goggle.png'
#firebase_storage_path = 'images/image.jpg'  # Adjust the path as needed

#blob = bucket.blob(firebase_storage_path)
#blob.upload_from_filename(local_image_path)

#print(f'Image uploaded to Firebase Storage at: {blob.public_url}')



##############################################################################################
######### download images from firebase and store it ####################################


#firebase_storage_path = 'images'
#bucket = storage.bucket()
# Local directory in the project to save downloaded images
#local_images_path = 'Images'
#os.makedirs(local_images_path, exist_ok=True)

# Download images from Firebase Storage to the local directory
#for blob in bucket.list_blobs(prefix=firebase_storage_path):
  #  blob_path = blob.name
  #  local_image_path = os.path.join(local_images_path, os.path.basename(blob_path))

    # Download image from Firebase Storage
   # blob.download_to_filename(local_image_path)

#print("Images downloaded and saved to the local 'images' directory with original file names")


#####################################################################################################################
############## Encode the images ################################################################




absolute_file_path = "/Users/omkar/PycharmProjects/object detection/venv/EncodeFile.p"

folderPath = 'images'
pathList = os.listdir(folderPath)

imgList = []
familyIds = []

# Filter out non-image files
image_extensions = ['.jpg', '.jpeg', '.png']
image_files = [f for f in pathList if os.path.splitext(f)[1].lower() in image_extensions]

for path in image_files:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    familyIds.append(os.path.splitext(path)[0])

print(familyIds)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding...")

# Check if there are images to encode
if imgList:
    encodeListKnown = findEncodings(imgList)
    encodeListKnownWithIds = [encodeListKnown, familyIds]
    print(encodeListKnown)
    print("Encoded")

    # Implementing error handling
    try:
        with open(absolute_file_path, 'wb') as file:
            pickle.dump(encodeListKnownWithIds, file)
        print(f"Encoded file saved at: {absolute_file_path}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("No images found to encode.")