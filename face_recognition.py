# face_recognition.py
import cv2, os
import numpy as np

# load mapping id->name
labels = {}
if os.path.exists("labels.csv"):
    with open("labels.csv", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                labels[int(parts[0])] = parts[1]

# load model
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    raise RuntimeError("cv2.face tidak tersedia. Install opencv-contrib-python")
recognizer.read("face-model.yml")

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

THRESHOLD = 60  # nilai eksperimental: semakin kecil -> lebih ketat (adjust sesuai dataset)
while True:
    ret, img = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (200,200))
        label, confidence = recognizer.predict(face_resized)  # confidence: nilai error (lebih kecil = cocok)
        if confidence < THRESHOLD:
            name = labels.get(label, f"User {label}")
            text = f"{name} ({confidence:.1f})"
        else:
            text = f"Unknown ({confidence:.1f})"
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow("Face Recognition - q to quit", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
