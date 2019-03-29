# coding: utf-8

import cv2
import numpy
import sys

if len(sys.argv) != 2:
	print("usage: python face.py image-file")
	sys.exit()

# load face database
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# read image
print(sys.argv[1])
img = cv2.imread(sys.argv[1])

# convert to gray image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face detect
faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor=1.1,
	minNeighbors=3,
	minSize=(30, 30),
	flags=cv2.CASCADE_SCALE_IMAGE
)

# add face border
for (x, y, w, h) in faces:
	cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("{} faces found".format(len(faces)), img)
cv2.waitKey(0)
