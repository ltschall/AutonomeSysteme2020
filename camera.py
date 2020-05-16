import face_recognition
import cv2
from tensorflow import keras
from keras.models import load_model
import numpy as np
import sys

# global variables
scale_factor = 4

# init expression face_recognition
emotions = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
label_map = dict((v,k) for k, v in emotions.items())
model = load_model('model_v6_23.hdf5')

# init camera
# video_capture = cv2.VideoCapture('http://192.168.178.51:4747/video?640x480')
input = sys.argv[1]
try:
    input = int(input)
except:
    pass
video_capture = cv2.VideoCapture(input)

# init some variables
frame_number = 0
face_locations = []
face_expressions = []
cropped = 0
while True:
    ret, frame = video_capture.read()
    if frame_number % 4 == 0:
        # face recognition
        small_frame = cv2.resize(frame, (0, 0), fx=1/scale_factor, fy=1/scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # expression recognition
        face_expressions = []
        for (top, right, bottom, left) in face_locations:
            face_image = frame[top*scale_factor:bottom*scale_factor, left*scale_factor:right*scale_factor]
            face_image = cv2.resize(face_image, (48, 48))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            cropped = face_image
            face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

            predicted_class = np.argmax(model.predict(face_image))
            face_expressions.append(label_map[predicted_class])

    frame_number += 1

    # graphical output
    for (top, right, bottom, left), face_expression in zip(face_locations, face_expressions):
        top *= scale_factor
        right *= scale_factor
        bottom *= scale_factor
        left *= scale_factor
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, face_expression, (left + 6, bottom -6), font, 1, (255, 255, 255), 1)

    # display resulting image
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    cv2.imshow('Video', frame)

    cv2.namedWindow('2', cv2.WINDOW_NORMAL)
    cv2.imshow('2', cropped)


    # break when 'q' is being pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
