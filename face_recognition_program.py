# face_recognition_program.py
import cv2
import face_recognition
import numpy as np
from reference_images import known_encodings

def recognize_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir BGR a RGB

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Desconocido"

        for i, match in enumerate(matches):
            if match:
                if i == 0:
                    name = "Barack Obama"
                elif i == 1:
                    name = "Elon Musk"
                elif i == 2:
                    name = "Cristiano Ronaldo"
                elif i == 3:
                    name = "Rafael Nadal"

        # Dibujar un rectángulo y el nombre en el rostro detectado
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

video_capture = cv2.VideoCapture(0)  # 0 para la cámara predeterminada

while True:
    ret, frame = video_capture.read()

    frame = recognize_faces(frame)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
