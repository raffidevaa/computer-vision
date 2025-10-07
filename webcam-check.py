# webcam-check.py
import cv2
cam = cv2.VideoCapture(0)   # ganti 0 jika punya lebih dari 1 kamera
if not cam.isOpened():
    raise RuntimeError("Tidak dapat membuka webcam")
while True:
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow("Webcam check - tekan q untuk keluar", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
