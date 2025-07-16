# from sklearn.neighbors import KNeighborsClassifier
# import cv2
# import pickle
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime

# from win32com.client import Dispatch

# def speak(str1):
#     speak=Dispatch(("SAPI.SpVoice"))
#     speak.Speak(str1)

# video = cv2.VideoCapture(0)
# facedetect=cv2.CascadeClassifier('new/haarcascade_frontalface_default.xml')

# with open('new/names.pkl', 'rb') as w:
#     LABELS = pickle.load(w)
# with open('new/faces_data.pkl', 'rb') as f:
#     FACES = pickle.load(f)

# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES,LABELS)

# COL_NAMES=['NAME',  'TIME']

# while True:
#     ret, frame = video.read()
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces=facedetect.detectMultiScale(gray,1.3, 5)
#     for(x,y,p,q) in faces:
#         crop_img= frame[y:y+q, x:x+p,:]
#         resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
#         result=knn.predict(resized_img) 
#         ts=time.time()
#         date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
#         exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
#         cv2.rectangle(frame, (x, y), (x+p, y+q), (0, 0, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+p, y+q), (60, 60, 255), 1)
#         cv2.rectangle(frame, (x, y-45), (x+p, y), (50, 50, 255), -1)
#         cv2.putText(frame,str(result[0]),(x,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
#         cv2.rectangle(frame,(x,y),(x+p, y+q), (40,40,255),1)
#         attendance=[str(result[0]),str(timestamp)]
#     cv2.imshow("Frame", frame)
#     k = cv2.waitKey(1)
#     if k == ord('o'):
#         speak("Your Attendance Are Taken..")
#         time.sleep(3)
#         if exist:
#             with open("Attendance/Attendance_" + date + ".csv","a") as csvfile:
#                 writer=csv.writer(csvfile)
#                 writer.writerow(attendance)
#             csvfile.close()
#         else:
#             with open("Attendance/Attendance_" + date + ".csv","a") as csvfile:
#                 writer=csv.writer(csvfile)
#                 writer.writerow(COL_NAMES)
#                 writer.writerow(attendance)
#             csvfile.close()
#     if k == ord('l'):
#         break
# video.release()
# cv2.destroyAllWindows()


import os
import pickle
import numpy as np
import cv2
from datetime import datetime
import time
import csv
from sklearn.neighbors import KNeighborsClassifier
from win32com.client import Dispatch


def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

# === Ensure directory exists ===
os.makedirs("new", exist_ok=True)
os.makedirs("Attendance", exist_ok=True)

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

# === Attendance Mode ===
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('new/haarcascade_frontalface_default.xml')
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_crop, (100, 100)).flatten().reshape(1, -1)

        # crop_img = frame[y:y + h, x:x + w, :]
        # resized_img = cv2.resize(crop_img, (100, 100))
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (153, 51, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (153, 51, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        attendance = [str(output[0]), str(timestamp)]

    cv2.imshow("Attendance", frame)
    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Your Attendance Is Taken")
        time.sleep(2)
        with open("Attendance/Attendance_" + date + ".csv", "+a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)
            writer.writerow(attendance)
    if k == ord('l'):
        break

video.release()
cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import os
# import csv
# import time
# from datetime import datetime
# from win32com.client import Dispatch
# from tensorflow.keras.models import load_model

# def speak(text):
#     speak = Dispatch("SAPI.SpVoice")
#     speak.Speak(text)

# video = cv2.VideoCapture(0)
# facedetect = cv2.CascadeClassifier(r'C:\Users\jeetender singh\OneDrive\Desktop\new\haarcascade_frontalface_default.xml')

# # Load CNN model
# model = load_model('path_to_your_trained_model.h5')

# COL_NAMES = ['NAME', 'TIME']

# while True:
#     ret, frame = video.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
#     for (x, y, w, h) in faces:
#         crop_img = frame[y:y+h, x:x+w, :]
#         resized_img = cv2.resize(crop_img, (50, 50))  # Resize image to match model input size
#         resized_img = resized_img.astype('float') / 255.0  # Normalize pixel values
#         resized_img = np.expand_dims(resized_img, axis=0)  # Add batch dimension
        
#         # Make prediction using the CNN model
#         prediction = model.predict(resized_img)
#         predicted_label = np.argmax(prediction)
        
#         ts = time.time()
#         date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
#         exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
#         cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
#         cv2.putText(frame, str(predicted_label), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        
#         attendance = [str(predicted_label), str(timestamp)]
        
#     cv2.imshow("Frame", frame)
#     k = cv2.waitKey(1)
    
#     if k == ord('o'):
#         speak("Your Attendance has been taken.")
#         time.sleep(3)
        
#         if exist:
#             with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(attendance)
#         else:
#             with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(COL_NAMES)
#                 writer.writerow(attendance)
                
#     if k == ord('l'):
#         break

# video.release()
# cv2.destroyAllWindows()

# import csv
# from sklearn.neighbors import KNeighborsClassifier
# import cv2
# import pickle
# import numpy as np
# import os
# import time
# from datetime import datetime
# from win32com.client import Dispatch

# def speak(str1):
#     speak=Dispatch(("SAPI.SpVoice"))
#     speak.Speak(str1)

# video=cv2.VideoCapture(0)
# facedetect=cv2.CascadeClassifier('new/haarcascade_frontalface_default.xml')

# with open('new/names.pkl', 'rb') as f:
#     LABELS=pickle.load(f)
# with open('new/faces_data.pkl', 'rb') as f:
#     FACES=pickle.load(f)

# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# COL_NAMES = ['NAME', 'TIME']

# attendance_captured = False  # Flag to indicate whether attendance has been captured

# while not attendance_captured:
#     ret,frame=video.read()
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces=facedetect.detectMultiScale(gray, 1.3 ,5)
#     for (x,y,w,h) in faces:
#         crop_img=frame[y:y+h, x:x+w, :]
#         resized_img=cv2.resize(crop_img, (100,100)).flatten().reshape(1,-1)
#         output=knn.predict(resized_img)
#         ts=time.time()
#         date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
#         exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (153,51,255), 1)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(153,51,255),2)
#         cv2.rectangle(frame,(x,y-40),(x+w,y),(153,51,255),-1)
#         cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (153,51,255), 1)
#         attendance=[str(output[0]), str(timestamp)]

#         # Attendance file handling
#         if exist:
#             with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                 writer=csv.writer(csvfile)
#                 writer.writerow(attendance)
#         else:
#             with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
#                 writer=csv.writer(csvfile)
#                 writer.writerow(COL_NAMES)
#                 writer.writerow(attendance)

#         attendance_captured = True  # Set flag to True to break out of the loop after capturing attendance

#         # Count students present
#         def count_students_present(csv_file):
#             students_present = set()  # Using a set to store unique student IDs or names
#             with open(csv_file, 'r') as file:
#                 reader = csv.reader(file)
#                 next(reader)  # Skip header row
#                 for row in reader:
#                     if row:  # Check if the row is not empty
#                         student_name = row[0]  # Assuming student name is in the first column
#                         students_present.add(student_name)
#             return len(students_present)

#         # Usage
#         csv_file_path = "Attendance/Attendance_" + date + ".csv"  # Provide the correct path to your CSV file
#         num_students_present = count_students_present(csv_file_path)
#         print("Total number of students present:", num_students_present)

#     cv2.imshow("Frame",frame)
#     k=cv2.waitKey(1)
#     if k==ord('o'):
#         speak("Your Attendance Are Taken..")
#         time.sleep(3)
#     if k==ord('l'):
#         break

# video.release()
# cv2.destroyAllWindows()


# import csv
# from sklearn.neighbors import KNeighborsClassifier
# import cv2
# import pickle
# import numpy as np
# import os
# import time
# from datetime import datetime
# from win32com.client import Dispatch

# def speak(str1):
#     speak=Dispatch(("SAPI.SpVoice"))
#     speak.Speak(str1)

# # Function to count students present
# def count_students_present(csv_file):
#     students_present = set()  # Using a set to store unique student IDs or names
#     with open(csv_file, 'r') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip header row
#         for row in reader:
#             if row:  # Check if the row is not empty
#                 student_name = row[0]  # Assuming student name is in the first column
#                 students_present.add(student_name)
#     return len(students_present)

# video=cv2.VideoCapture(0)
# facedetect=cv2.CascadeClassifier('new/haarcascade_frontalface_default.xml')

# with open('new/names.pkl', 'rb') as f:
#     LABELS=pickle.load(f)
# with open('new/faces_data.pkl', 'rb') as f:
#     FACES=pickle.load(f)

# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# COL_NAMES = ['NAME', 'TIME']

# attendance_captured = False  # Flag to indicate whether attendance has been captured

# while not attendance_captured:
#     ret,frame=video.read()
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces=facedetect.detectMultiScale(gray, 1.3 ,5)
#     for (x,y,w,h) in faces:
#         crop_img=frame[y:y+h, x:x+w, :]
#         resized_img=cv2.resize(crop_img, (100,100)).flatten().reshape(1,-1)
#         output=knn.predict(resized_img)
#         ts=time.time()
#         date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
#         exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (153,51,255), 1)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(153,51,255),2)
#         cv2.rectangle(frame,(x,y-40),(x+w,y),(153,51,255),-1)
#         cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (153,51,255), 1)
#         attendance=[str(output[0]), str(timestamp)]

#         # Attendance file handling
#         if exist:
#             with open("Attendance/Attendance_" + date + ".csv", "r") as csvfile:
#                 reader = csv.reader(csvfile)
#                 existing_data = list(reader)
#                 if len(existing_data) <= 1:  # Check if only header row exists
#                     with open("Attendance/Attendance_" + date + ".csv", "a", newline="") as csvfile:
#                         writer = csv.writer(csvfile)
#                         writer.writerow(COL_NAMES)
#                         writer.writerow(attendance)
#                         attendance_captured = True
#         else:
#             with open("Attendance/Attendance_" + date + ".csv", "a", newline="") as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(COL_NAMES)
#                 writer.writerow(attendance)
#                 attendance_captured = True

#         # Check for 'o' key press to capture attendance
#         k=cv2.waitKey(1)
#         if k==ord('o'):
#             speak("Your Attendance Are Taken..")
#             time.sleep(3)  # Wait for 3 seconds before displaying the total number of students present
#             csv_file_path = "Attendance/Attendance_" + date + ".csv"
#             num_students_present = count_students_present(csv_file_path)
#             print("Total number of students present:", num_students_present)
#             break  # Break out of the loop after displaying the total number of students present

#     cv2.imshow("Frame",frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('l'):
#         break

# video.release()
# cv2.destroyAllWindows()

# import csv
# from sklearn.neighbors import KNeighborsClassifier
# import cv2
# import pickle
# import numpy as np
# import os
# import time
# from datetime import datetime
# from win32com.client import Dispatch

# def speak(str1):
#     speak=Dispatch(("SAPI.SpVoice"))
#     speak.Speak(str1)

# # Function to count students present
# def count_students_present(csv_file):
#     students_present = set()  # Using a set to store unique student IDs or names
#     with open(csv_file, 'r') as file:
#         reader = csv.reader(file)
#         next(reader)  # Skip header row
#         for row in reader:
#             if row:  # Check if the row is not empty
#                 student_name = row[0]  # Assuming student name is in the first column
#                 students_present.add(student_name)
#     return len(students_present)

# video=cv2.VideoCapture(0)
# facedetect=cv2.CascadeClassifier('new/haarcascade_frontalface_default.xml')

# with open('new/names.pkl', 'rb') as f:
#     LABELS=pickle.load(f)
# with open('new/faces_data.pkl', 'rb') as f:
#     FACES=pickle.load(f)

# knn=KNeighborsClassifier(n_neighbors=5)
# knn.fit(FACES, LABELS)

# COL_NAMES = ['NAME', 'TIME']

# # Keep track of whether attendance has been captured in the current session
# attendance_captured = False

# while True:  # Loop indefinitely
#     ret,frame=video.read()
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces=facedetect.detectMultiScale(gray, 1.3 ,5)
#     for (x,y,w,h) in faces:
#         crop_img=frame[y:y+h, x:x+w, :]
#         resized_img=cv2.resize(crop_img, (100,100)).flatten().reshape(1,-1)
#         output=knn.predict(resized_img)
#         ts=time.time()
#         date=datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
#         timestamp=datetime.fromtimestamp(ts).strftime("%H:%M-%S")
#         exist=os.path.isfile("Attendance/Attendance_" + date + ".csv")
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (153,51,255), 1)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(153,51,255),2)
#         cv2.rectangle(frame,(x,y-40),(x+w,y),(153,51,255),-1)
#         cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (153,51,255), 1)
#         attendance=[str(output[0]), str(timestamp)]

#         # Attendance file handling
#         if exist and not attendance_captured:  # Only write attendance if it hasn't been captured yet in this session
#              with open("Attendance/Attendance_" + date + ".csv", "r") as csvfile:
#                 reader = csv.reader(csvfile)
#                 existing_data = list(reader)
#                 if len(existing_data) <= 1:  # Check if only header row exists
#                     with open("Attendance/Attendance_" + date + ".csv", "a", newline="") as csvfile:
#                         writer = csv.writer(csvfile)
#                         writer.writerow(COL_NAMES)
#                         writer.writerow(attendance)
#                         attendance_captured = True
#         else:                
#             with open("Attendance/Attendance_" + date + ".csv", "a", newline="") as csvfile:
#                 writer = csv.writer(csvfile)
#                 writer.writerow(COL_NAMES)  # Write column names if file is empty
#                 writer.writerow(attendance)
#                 attendance_captured = True

#         # Check for 'o' key press to capture attendance
#         k=cv2.waitKey(1)
#         if k==ord('o') and not attendance_captured:
#             speak("Your Attendance Are Taken..")
#             time.sleep(3)  # Wait for 3 seconds before displaying the total number of students present
#             csv_file_path = "Attendance/Attendance_" + date + ".csv"
#             num_students_present = count_students_present(csv_file_path)
#             print("Total number of students present:", num_students_present)
#             break  # Break out of the loop after displaying the total number of students present

#     cv2.imshow("Frame",frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('l'):
#         break

# video.release()
# cv2.destroyAllWindows()

