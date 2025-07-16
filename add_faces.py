
# import os
# import pickle
# import numpy as np
# import cv2
# from datetime import datetime
# import time
# import csv
# from sklearn.neighbors import KNeighborsClassifier
# from win32com.client import Dispatch

# def speak(str1):
#     speak = Dispatch("SAPI.SpVoice")
#     speak.Speak(str1)

# # === Ensure directory exists ===
# os.makedirs("new", exist_ok=True)
# os.makedirs("Attendance", exist_ok=True)

# # === Safely load pickle files or initialize ===
# labels_path = 'new/names.pkl'
# faces_path = 'new/faces_data.pkl'

# if os.path.exists(labels_path) and os.path.getsize(labels_path) > 0:
#     with open(labels_path, 'rb') as f:
#         LABELS = pickle.load(f)
# else:
#     LABELS = []

# if os.path.exists(faces_path) and os.path.getsize(faces_path) > 0:
#     with open(faces_path, 'rb') as f:
#         FACES = pickle.load(f)
# else:
#     FACES = np.empty((0, 10000))

# # === Face Capture ===
# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier('new/haarcascade_frontalface_default.xml')
# faces_data = []
# i = 0

# name = input("Enter Your Name: ")

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y + h, x:x + w, :]
#         gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
#         resized_img = cv2.resize(gray_crop, (100, 100))
#         if len(faces_data) <= 100 and i % 10 == 0:
#             faces_data.append(resized_img)
#         i += 1
#         cv2.putText(frame, str(len(faces_data)), (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (102, 255, 102), 1)
#     cv2.imshow("Face Register", frame)
#     k = cv2.waitKey(1)
#     if k == ord('l') or len(faces_data) == 100:
#         break
# video.release()
# cv2.destroyAllWindows()

# faces_data = np.asarray(faces_data)
# faces_data = faces_data.reshape(100, -1)

# # === Update and Save Pickle Files ===
# FACES = np.append(FACES, faces_data, axis=0)
# LABELS.extend([name] * 100)

# with open(faces_path, 'wb') as f:
#     pickle.dump(FACES, f)

# with open(labels_path, 'wb') as f:
#     pickle.dump(LABELS, f)

# print(f"Face data saved successfully for {name}")

# import cv2
# import pickle
# import numpy as np
# import os
# video=cv2.VideoCapture(0)
# facedetect=cv2.CascadeClassifier('new/haarcascade_frontalface_default.xml')

# faces_data=[]

# i=0

# name=input("Enter Your Name: ")

# while True:
#     ret,frame=video.read()
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces=facedetect.detectMultiScale(gray, 1.3 ,5)
#     for (x,y,w,h) in faces:
#         crop_img=frame[y:y+h, x:x+w, :]
#         resized_img=cv2.resize(crop_img, (100,100))
#         if len(faces_data)<=100 and i%10==0:
#             faces_data.append(resized_img)
#         i=i+1
#         cv2.putText(frame, str(len(faces_data)), (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (102,255,102), 1)
#     cv2.imshow("Frame",frame)
#     k=cv2.waitKey(1)
#     if k==ord('l') or len(faces_data)==100:
#         break
# video.release()
# cv2.destroyAllWindows()

# faces_data=np.asarray(faces_data)
# faces_data=faces_data.reshape(100, -1)


# import os

# labels_path = 'new/names.pkl'
# faces_path = 'new/faces_data.pkl'

# if os.path.exists(labels_path) and os.path.getsize(labels_path) > 0:
#     with open(labels_path, 'rb') as f:
#         LABELS = pickle.load(f)
# else:
#     print("Error: names.pkl is empty or missing.")
#     LABELS = []  # or some default fallback
#     exit()

# if os.path.exists(faces_path) and os.path.getsize(faces_path) > 0:
#     with open(faces_path, 'rb') as f:
#         FACES = pickle.load(f)
# else:
#     print("Error: faces_data.pkl is empty or missing.")
#     FACES = []
#     exit()



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



# video = cv2.VideoCapture(0)
# # facedetect = cv2.CascadeClassifier('C:\Users\jeetender singh\OneDrive\Desktop\new\haarcascade_frontalface_default.xml')
# facedetect = cv2.CascadeClassifier(r'C:\Users\jeetender singh\OneDrive\Desktop\new\haarcascade_frontalface_default.xml')


# faces_data = []
# labels = []

# name = input("Enter Your Name: ")

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
#     for (x, y, w, h) in faces:
#         face_img = frame[y:y+h, x:x+w]
#         face_img = cv2.resize(face_img, (224, 224))  # Resize the image to fit the CNN input
#         face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (since MobileNetV2 expects RGB)
#         face_img = img_to_array(face_img)
#         face_img = preprocess_input(face_img)
        
#         faces_data.append(face_img)
#         labels.append(name)
        
#         cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        
#     cv2.imshow("Frame", frame)
#     k = cv2.waitKey(1)
#     if k == ord('l') or len(faces_data) == 100:
#         break

# video.release()
# cv2.destroyAllWindows()

# # Convert lists to arrays
# faces_data = np.array(faces_data)
# labels = np.array(labels)

# # Save the arrays to files
# # np.save('new:/faces_data.npy', faces_data)
# # np.save('new:/labels.npy', labels)

# np.save('C:/Users/jeetender singh/OneDrive/Desktop/new/faces_data.npy', faces_data)
# np.save('C:/Users/jeetender singh/OneDrive/Desktop/new/labels.npy', labels)

# === Safely load pickle files or initialize ===
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




# # === Attendance Mode ===
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier('new/haarcascade_frontalface_default.xml')
# COL_NAMES = ['NAME', 'TIME']

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w]
#         gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
#         resized_img = cv2.resize(gray_crop, (100, 100)).flatten().reshape(1, -1)

#         # crop_img = frame[y:y + h, x:x + w, :]
#         # resized_img = cv2.resize(crop_img, (100, 100))
#         output = knn.predict(resized_img)
#         ts = time.time()
#         date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
#         exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

#         cv2.rectangle(frame, (x, y), (x + w, y + h), (153, 51, 255), 2)
#         cv2.rectangle(frame, (x, y - 40), (x + w, y), (153, 51, 255), -1)
#         cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

#         attendance = [str(output[0]), str(timestamp)]

#     #cv2.imshow("Attendance", frame)
#     k = cv2.waitKey(1)
#     # if k == ord('o'):
#     #     speak("Your Attendance Is Taken")
#     #     time.sleep(2)
#     #     with open("Attendance/Attendance_" + date + ".csv", "+a", newline='') as csvfile:
#     #         writer = csv.writer(csvfile)
#     #         if not exist:
#     #             writer.writerow(COL_NAMES)
#     #         writer.writerow(attendance)
#     if k == ord('l'):
#         break

video.release()
cv2.destroyAllWindows()