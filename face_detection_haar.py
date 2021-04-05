import cv2 as cv

# Reading & Resizing image
img = cv.imread(r"data\images\humans.jpeg")
img = cv.resize(img, (img.shape[1]//2, img.shape[0]//2), cv.INTER_AREA)
cv.imshow("Person", img)
cv.waitKey(delay=60)

# Converting color image to grayscale image as haar cascade needs grayscale image.
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Gray", gray_img)
cv.waitKey(delay=60)

# Loading Haar Cascade & detecting faces
haar_cascade = cv.CascadeClassifier(r"data\cascades\haarcascade_face.xml")
face_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3)
print(f"Number of faces found: {len(face_rect)}")

# Creating rectangles around faces found
for x, y, w, h in face_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
cv.imshow("Faces Found", img)
cv.waitKey(0)