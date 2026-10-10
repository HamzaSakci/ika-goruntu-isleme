import cv2
import numpy as np
import os

def find_image_center(image):
    """
    Verilen bir görüntünün merkez koordinatlarını (cx, cy) hesaplar.
    
    Argümanlar:
        image (numpy.ndarray): OpenCV ile okunmuş görüntü matrisi.
        
    Dönüş:
        tuple: (cx, cy) şeklinde merkez piksel koordinatları (tam sayı).
    """
    # Görüntünün boyutlarını al (yükseklik, genişlik)
    h, w = image.shape[:2]
    
    # Merkezi bulmak için genişlik ve yüksekliği 2'ye tam böl (//)
    cx = w // 2
    cy = h // 2
    
    return (cx, cy)

if __name__ == "__main__":
    # Test için boş, siyah bir görüntü oluşturalım (Yükseklik: 480, Genişlik: 640)
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Fonksiyonu çağırarak merkezi bulalım
    merkez = find_image_center(test_img)
    print(f"640x480 boyutundaki görüntünün merkezi: {merkez}")
    
    # Görüntü üzerine kırmızı bir artı (cross) işareti çizelim
    # (0, 0, 255) -> OpenCV'de BGR formatı olduğu için Kırmızı demektir.
    cv2.drawMarker(test_img, merkez, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    
    # Sonucu 'outputs' klasörüne kaydedelim
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True) # Klasör yoksa oluştur
    
    output_path = os.path.join(output_dir, "01_merkez.png")
    ok, buf = cv2.imencode(".png", test_img)
    if ok:
        buf.tofile(output_path)
        print(f"Test görüntüsü başarıyla kaydedildi: {output_path}")
