# reference_images.py
import os
import face_recognition

image_folder = "imagenes"

# Nombre de las imágenes de las personas que deseas reconocer
personas = ["BarackObama.png", "ElonMusk.png", "CristianoRonaldo.png", "RafaelNadal.png"]

known_encodings = []

for imagen in personas:
    image_path = os.path.join(image_folder, imagen)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_encodings.append(encoding)
