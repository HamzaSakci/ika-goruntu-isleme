import cv2
import numpy as np
import glob
import os

def nothing(x):
    pass

# Trackbar penceresini oluştur
cv2.namedWindow('HSV Tuner')
cv2.resizeWindow('HSV Tuner', 400, 250)

# Başlangıç değerleri (Bizim kodda kullandığımız değerler)
cv2.createTrackbar('H Min', 'HSV Tuner', 5, 179, nothing)
cv2.createTrackbar('S Min', 'HSV Tuner', 100, 255, nothing)
cv2.createTrackbar('V Min', 'HSV Tuner', 100, 255, nothing)
cv2.createTrackbar('H Max', 'HSV Tuner', 25, 179, nothing)
cv2.createTrackbar('S Max', 'HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('V Max', 'HSV Tuner', 255, 255, nothing)

# Test edilecek görüntüyü yükle
dataset_dir = os.path.join(os.path.dirname(__file__), "..", "data", "duba_dataset", "test", "images")
image_files = glob.glob(os.path.join(dataset_dir, "*.jpg"))

if len(image_files) == 0:
    print("Görüntü bulunamadı!")
    exit()

# Görüntüyü Türkçe karakterleri destekleyecek şekilde oku
img_path = image_files[0]
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
img = cv2.resize(img, (640, 480)) # Ekrana sığması için yeniden boyutlandır
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print("HSV ayarlayıcı açıldı. Çıkmak için 'q' tuşuna basın.")

while True:
    # Trackbar değerlerini oku
    h_min = cv2.getTrackbarPos('H Min', 'HSV Tuner')
    s_min = cv2.getTrackbarPos('S Min', 'HSV Tuner')
    v_min = cv2.getTrackbarPos('V Min', 'HSV Tuner')
    h_max = cv2.getTrackbarPos('H Max', 'HSV Tuner')
    s_max = cv2.getTrackbarPos('S Max', 'HSV Tuner')
    v_max = cv2.getTrackbarPos('V Max', 'HSV Tuner')

    # Maske oluştur
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # Gürültü temizleme (Opsiyonel ama iyi sonuç verir)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Orijinal görüntü ile maskeyi birleştirip sonucu göster
    result = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('Orijinal', img)
    cv2.imshow('Maske (Siyah/Beyaz)', mask)
    cv2.imshow('Sonuc', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
