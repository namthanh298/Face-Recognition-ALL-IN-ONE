# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:53:26 2021

@author: HuyHoang
"""
# SECTION 1 - PRE PROCESS
# Include library
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
'''#Age detector workaround
args = {
    "face": "face_detector",
    "age": "age_detector",
    "confidence": 0.9
}'''
## Dataset for age dectect
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']
age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
# Include dataset for name label
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
##################################################################################################EMOTION_PREDICT_PREPARE#########
# Include dataset for emotion predict
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
#################################################################################################################################
#################################################################################################################################
#################################################NAME_RECOGINIZE_PREPARE#########################################################
# Create a list of images and a list of corresponding names
(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
	for subdir in dirs:
		names[id] = subdir
		subjectpath = os.path.join(datasets, subdir)
		for filename in os.listdir(subjectpath):
			path = subjectpath + '/' + filename
			lable = id
			images.append(cv2.imread(path, 0))
			lables.append(int(lable))
		id += 1
(width, height) = (130, 100)
# Create a Numpy array from the two lists above
(images, lables) = [np.array(lis) for lis in [images, lables]]
# OpenCV trains a model from the images
# NOTE FOR OpenCV2: remove '.face'
model2 = cv2.face.LBPHFaceRecognizer_create()
model2.train(images, lables)
###################################################################################################################################
###################################################################################################################################
# Haar Cascade
face_cascade = cv2.CascadeClassifier(haar_file)
'''# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# load our serialized age detector model from disk
print("[INFO] loading age detector model...")
prototxtPath = os.path.sep.join([args["age"], "age_deploy.prototxt"])
weightsPath = os.path.sep.join([args["age"], "age_net.caffemodel"])
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)'''
# Cam check
# Main process
cap=cv2.VideoCapture(0)
if not cap.isOpened:                   # Nếu webcam ko bật
    print('--(!)Error opening video capture')
    exit(0)
    
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x,y,h,w) in faces_detected:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        # Age & Gender
        age_gender_sample = frame[y:y+w, x:x+h].copy()
        blob = cv2.dnn.blobFromImage(age_gender_sample, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        ### Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender : " + gender)
        ### Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print("Age Range: " + age)
        ### Text
        overlay_text = "%s %s" % (gender, age)
        # Covert 2nd time
        roi_gray = gray[y:y+w,x:x+h]
        roi_gray = cv2.resize(roi_gray,(48,48))  
        image_pixels = image.img_to_array(roi_gray)
        image_pixels = np.expand_dims(image_pixels, axis = 0)
        image_pixels /= 255
        # Emotion
        e_predict = model.predict(image_pixels)
        max_index = np.argmax(e_predict)
        emotion =  ('Boast', 'Angry', 'Fear', 'Happy', 'Sad', 'Suprise', 'Neutral')
        emotion_prediction = emotion[max_index]
        cv2.putText(frame, emotion_prediction, (x-10, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(0,255,255), thickness=4)
        # Label Name
        n_predict = model2.predict(roi_gray)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        if n_predict[1]<170:
            cv2.putText(frame, '% s' %(names[n_predict[0]]), (x+120, y-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness = 4);
        else:
            cv2.putText(frame, 'not recognized',(x+120, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.putText(frame, overlay_text, (x+120, y-120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Emotion', frame)
        
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()