# face_create_dataset.py
import cv2, os

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

user_id = input("Masukkan user numeric id (contoh: 1): ").strip()
user_name = input("Masukkan nama (contoh: Alice): ").strip()

dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

# simpan mapping id->name
with open("labels.csv", "a", encoding="utf-8") as f:
    f.write(f"{user_id},{user_name}\n")

sample_count = 0
MAX_SAMPLES = 30

print("Persiapkan wajah, tekan 'q' untuk berhenti lebih awal.")
while True:
    ret, img = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        sample_count += 1
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (200,200))
        filename = os.path.join(dataset_path, f"User.{user_id}.{sample_count}.jpg")
        cv2.imwrite(filename, face_resized)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(img, f"#{sample_count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imshow("Create Dataset - q to quit", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if sample_count >= MAX_SAMPLES:
        break

print(f"Dataset selesai, total sampel: {sample_count}")
cam.release()
cv2.destroyAllWindows()
