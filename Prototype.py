# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:02:26 2021

@author: HuyHoang
"""
# Include library
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

#load model fer_model2 (chứa file .pb)
model = tf.keras.models.load_model('fer_model2')             # Thay đường dẫn đến folder fer_model2 vào
# Chuyển .pb sang .json (cho opencv hỗ trợ)
model_json = model.to_json()
# load lại model từ file .json
model = model_from_json(model_json)
# load weights đã train
model.load_weights('fer_model2/fer_model2.h5')             # Thay đường dẫn đến file fer_model2.h5 vào
# sử dụng haar_cascade để nhận diện face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Code demo test cam & face
cap=cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, 
                                                        minSize=(30, 30)) 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 7)
    
    cv2.imshow("Faces found", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Main process
cap=cv2.VideoCapture(0)
if not cap.isOpened:                   # Nếu webcam ko bật
    print('--(!)Error opening video capture')
    exit(0)

while (True):             # Nếu webcam bật
    ret, frame = cap.read()               # đọc từng frame trong video stream

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       # chuyển frame sang grayscale
# Nhận diện face bằng haar_cascade
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x,y,w,h) in faces_detected:
        # Vẽ bounding box cho face
        cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,0), 7)
# NHẬN DIỆN CẢM XÚC
        # Phân vùng ảnh grayscale chứa face
        roi_gray = gray[y:y+w,x:x+h]
        roi_gray = cv2.resize(roi_gray,(48,48))          # target_size = (48,48) để fit với input của CNN đang dùng
        # Đưa roi_gray về dạng array và chuẩn hóa, để model predict
        image_pixels = image.img_to_array(roi_gray)
        image_pixels = np.expand_dims(image_pixels, axis = 0)
        image_pixels /= 255
        # Thực hiện model predict
        predictions = model.predict(image_pixels)
        max_index = np.argmax(predictions)      #  Lấy ra chỉ mục có xác suất dự đoán cao nhất
        
        # Từ max_index suy ra loại cảm xúc được phân loại
        emotion =  ('Boast', 'Angry', 'Fear', 'Happy', 'Sad', 'Suprise', 'Neutral')
        emotion_prediction = emotion[max_index]

        # Gán label text cho bounding box của face
        cv2.putText(frame, emotion_prediction, (x-10, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(0,255,255), thickness=4)
        #cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)  

        #resized_image = cv2.resize(x, (1000, 700))
    cv2.imshow('Emotion', frame)
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()