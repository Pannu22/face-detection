import os
from pathlib import Path

import cv2 as cv
import numpy as np

train_data_dir = Path(r"data\faces\train")
people = [x.name for x in train_data_dir.iterdir() if x.is_dir()]
print(f"People's list: {people}")

haar_cascade = cv.CascadeClassifier(r"data\cascades\haarcascade_face.xml")

features = []
labels = []
labels_mapping = {}

def create_train_data(training_img, img_label):
    """
    Extracting Face Region of Interest for images & creating features.
    Input  :
        training_img  : Training image with BGR format.
        img_label     : Label encoded for image.
    Return :
        features_list : List of faces.
        labels_list   : List of labels encoded for faces.
    """
    features_list = []
    labels_list = []

    # Converting color image into grayscale image & detecting faces
    gray_img = cv.cvtColor(training_img, cv.COLOR_BGR2GRAY, cv.INTER_AREA)
    faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4)

    # Extracting faces from image
    for x,y,w,h in faces_rect:
        faces_roi = gray_img[y:y+h, x:x+w]
        features_list.append(faces_roi)
        labels_list.append(img_label)

    return features_list, labels_list

for person_label, person in enumerate(people):
    labels_mapping[person_label] = person
    person_imgs_dir_path = train_data_dir / Path(person)

    for each_img in person_imgs_dir_path.iterdir():
        img = cv.imread(str(each_img))
        face_features, face_labels = create_train_data(img, person_label)
        features.extend(face_features)
        labels.extend(face_labels)

# Converting features & there labels to a numpy array
features = np.array(features, dtype="object")
labels = np.array(labels)

# Training face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

# Saving traineed face recognizer, features & labels
face_recognizer.save(r"model/face_recognizer_trainer.yml")
np.save(r"model\features.npy", features)
np.save(r"model\labels.npy", labels)