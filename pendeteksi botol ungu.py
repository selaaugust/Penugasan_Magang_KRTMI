import cv2
import numpy as np
import time

# Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Buka kamera
cap = cv2.VideoCapture(0)
time.sleep(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------
    # 1. DETEKSI WARNA UNGU (BOTOL)
    # -------------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Range warna ungu (bisa disesuaikan)
    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([155, 255, 255])

    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 1000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(frame, "Botol Ungu", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    # -------------------------
    # 2. DETEKSI WAJAH
    # -------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 0), 3)
        cv2.putText(frame, "Wajah", (fx, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Tampilkan hasil dalam 1 window
    cv2.imshow("Deteksi Wajah & Botol Ungu", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
