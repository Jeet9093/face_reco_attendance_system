
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
import pickle
import time
from datetime import datetime

labels_path = 'new/names.pkl'
faces_path = 'new/faces_data.pkl'

if os.path.exists(labels_path) and os.path.getsize(labels_path) > 0:
    with open(labels_path, 'rb') as f:
        LABELS = pickle.load(f)
else:
    LABELS = []

if os.path.exists(faces_path) and os.path.getsize(faces_path) > 0:
    with open(faces_path, 'rb') as f:
        FACES = pickle.load(f)
else:
    FACES = np.empty((0, 10000))

# === Face Capture ===
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('new/haarcascade_frontalface_default.xml')
faces_data = []
i = 0

name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_crop, (100, 100))
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (102, 255, 102), 1)
    cv2.imshow("Face Register", frame)
    k = cv2.waitKey(1)
    if k == ord('l') or len(faces_data) == 100:
        break
video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

# === Update and Save Pickle Files ===
FACES = np.append(FACES, faces_data, axis=0)
LABELS.extend([name] * 100)

with open(faces_path, 'wb') as f:
    pickle.dump(FACES, f)

with open(labels_path, 'wb') as f:
    pickle.dump(LABELS, f)

print(f"Face data saved successfully for {name}")

video.release()
cv2.destroyAllWindows()