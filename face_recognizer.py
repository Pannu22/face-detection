import cv2 as cv
import json

# Reading labels encoding of trained model.
with open(r"model\face_recognizer_label_mapping.json", "r") as labels_mapping_json:
    labels_mapping = json.load(labels_mapping_json)
    labels_mapping_json.close()

test_img = cv.imread(r"data\faces\val\elton_john\1.jpg")

# Initiating face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r"model\face_recognizer_trained.yml")

# Preprocessing image for face recognization
haar_cascade = cv.CascadeClassifier(r"data\cascades\haarcascade_face.xml")
gray_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 4)

for x, y, w, h in faces_rect:
    face_roi = gray_img[y:y+h, x:x+w]

    label, confidence = face_recognizer.predict(face_roi)
    cv.putText(test_img, f"{labels_mapping[str(label)]}-{confidence}", (20, 20), fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=1.0, color=(0, 0, 255), thickness=2)
    cv.rectangle(test_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv.imshow("Faces Recognized", test_img)
cv.waitKey(0)
