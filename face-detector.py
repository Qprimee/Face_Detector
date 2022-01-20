import cv2

# load some pre-trained date on face fronetals from opencv (haar cascade algoritm)
trained_face_date = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect face in
img = cv2.imread('10.jpg')

# Must convert to grayscal
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect face
face_coordinates = trained_face_date.detectMultiScale(grayscaled_img)

# Drow erctangles around the face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)

#
cv2.imshow('Face Detector', img)
cv2.waitKey()