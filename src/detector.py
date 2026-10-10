import cv2
import numpy as np
import os
import glob

# Duba (turuncu/kırmızı) HSV aralıkları.
# Gölgedeki kısımlar H=0-5 civarına (kırmızıya) kayar; kırmızı H ekseninde
# 0'da başa sardığı için 170-179 aralığı da eklenir.
HSV_RANGES = [
    (np.array([0, 150, 30]), np.array([25, 255, 255])),
    (np.array([170, 150, 30]), np.array([179, 255, 255])),
]
MIN_AREA = 200


def read_image_utf8(path):
    """Türkçe karakter içeren yollardaki görüntüleri güvenle okur."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def detect_duba(img):
    """
    Görüntüdeki dubayı HSV maskesiyle bulur.
    Dönüş: (cx, cy, (x, y, w, h), mask) -> duba yoksa (None, None, None, mask)

    Not: Dubanın beyaz şeritleri turuncu maskeyi parçalar. Dikey uzun bir
    kernel ile CLOSE yaparak parçaları birleştiriyoruz, merkez olarak da
    bounding box merkezini alıyoruz.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros(img.shape[:2], np.uint8)
    for lo, hi in HSV_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)

    # Küçük gürültüleri sil
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Beyaz şeritlerin yarattığı dikey boşlukları kapat
    h = img.shape[0]
    k_h = max(15, h // 12)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, k_h)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_AREA:
        return None, None, None, mask

    x, y, bw, bh = cv2.boundingRect(largest)
    cx, cy = x + bw // 2, y + bh // 2
    return cx, cy, (x, y, bw, bh), mask


def load_yolo_center(label_path, w, h):
    """
    YOLO etiketinden (ilk nesne) gerçek merkezi ve kutuyu piksel olarak döndürür.
    İki format desteklenir:
      - bbox:    sınıf xc yc w h            (5 değer)
      - poligon: sınıf x1 y1 x2 y2 x3 y3 ... (Roboflow segmentasyon çıktısı)
    """
    if not os.path.exists(label_path):
        return None
    with open(label_path) as f:
        parts = f.readline().strip().split()
    vals = [float(v) for v in parts[1:]]
    if len(vals) == 4:
        xc, yc, bw, bh = vals
        x1, y1, x2, y2 = (xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h
    elif len(vals) >= 6:
        xs, ys = vals[0::2], vals[1::2]
        x1, y1, x2, y2 = min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h
    else:
        return None
    return int((x1 + x2) / 2), int((y1 + y2) / 2), (int(x1), int(y1), int(x2), int(y2))


def process_duba_detection(image_path, output_path, label_path=None):
    img = read_image_utf8(image_path)
    if img is None:
        print(f"Hata: Görüntü okunamadı -> {image_path}")
        return

    h, w = img.shape[:2]
    cx0, cy0 = w // 2, h // 2
    out = img.copy()

    cv2.drawMarker(out, (cx0, cy0), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(out, "Goruntu Merkezi", (cx0 - 60, cy0 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cx, cy, box, _ = detect_duba(img)

    if cx is not None:
        x, y, bw, bh = box
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 6, (0, 255, 0), -1)
        dx, dy = cx - cx0, cy - cy0
        cv2.line(out, (cx0, cy0), (cx, cy), (0, 255, 255), 2)
        cv2.putText(out, f"dx: {dx} px, dy: {dy} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        print(f"Duba bulundu! Merkez: ({cx}, {cy}) | Fark: dx={dx}, dy={dy}")
    else:
        print("Görüntüde duba bulunamadı.")

    # Etiketteki gerçek merkezle karşılaştır (kırmızı)
    gt = load_yolo_center(label_path, w, h) if label_path else None
    if gt is not None:
        gx, gy, (x1, y1, x2, y2) = gt
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(out, (gx, gy), 6, (0, 0, 255), -1)
        if cx is not None:
            err = float(np.hypot(cx - gx, cy - gy))
            cv2.putText(out, f"Etiket(kirmizi) hata: {err:.1f} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"Etiket merkezi: ({gx}, {gy}) | Piksel hatası: {err:.1f}")

    ok, buf = cv2.imencode(".png", out)
    if ok:
        buf.tofile(output_path)
        print(f"Çıktı kaydedildi: {output_path}")


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..", "data", "duba_dataset", "test")
    image_files = glob.glob(os.path.join(base, "images", "*.jpg"))

    if image_files:
        test_image = image_files[0]
        label = os.path.join(base, "labels", os.path.splitext(os.path.basename(test_image))[0] + ".txt")
        out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Test ediliyor: {os.path.basename(test_image)}")
        process_duba_detection(test_image, os.path.join(out_dir, "02_tespit.png"), label)
    else:
        print("Hata: data/duba_dataset/test/images klasöründe resim bulunamadı.")


