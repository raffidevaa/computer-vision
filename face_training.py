# face_training.py
import cv2, os
import numpy as np

dataset_path = "dataset"
image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith('.jpg')]
faces = []
labels = []

for file in image_files:
    path = os.path.join(dataset_path, file)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    # ambil label dari filename: User.<id>.<n>.jpg
    name = os.path.splitext(file)[0]
    parts = name.split(".")
    try:
        label = int(parts[1])
    except:
        continue
    faces.append(img)
    labels.append(label)

if len(faces) == 0:
    raise RuntimeError("Tidak ada data pada folder dataset/ — jalankan face_create_dataset.py dulu")

# buat recognizer LBPH
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    raise RuntimeError("cv2.face tidak tersedia. Install opencv-contrib-python: pip install opencv-contrib-python")

recognizer.train(faces, np.array(labels))
recognizer.write("face-model.yml")
print("Training selesai. Model disimpan ke face-model.yml")
