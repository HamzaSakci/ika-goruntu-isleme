import cv2
import numpy as np
import csv
import time
import os
from detector import detect_duba

def run_video_tracking(video_source=0, output_csv="outputs/tracking_log.csv"):
    """
    Video veya web kamerası üzerinden canlı olarak dubayı takip eder,
    komutları üretir ve bir CSV dosyasına kaydeder (Loglama).
    video_source: 0 (webcam) veya 'video.mp4' dosya yolu.
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Hata: Video kaynağı açılamadı!")
        return

    # CSV Log dosyasını hazırla (Word dosyasındaki 'Log yoksa hata üretilemez' kuralı)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Timestamp", "dx", "dy", "ex", "ey", "Command"])
        
        frame_count = 0
        print("Video işleme başladı. Çıkmak için 'q' tuşuna basın.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            timestamp = time.strftime("%H:%M:%S")
            
            h, w = frame.shape[:2]
            cx0, cy0 = w // 2, h // 2
            
            # Tespit (detector.py ile ortak)
            cx, cy, box, _ = detect_duba(frame)
            
            # Görüntü merkezini çiz
            cv2.drawMarker(frame, (cx0, cy0), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
            
            command = "NESNE YOK"
            dx, dy, ex, ey = 0, 0, 0.0, 0.0
            
            if cx is not None:
                x, y, bw, bh = box
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.line(frame, (cx0, cy0), (cx, cy), (0, 255, 255), 2)
                
                dx = cx - cx0
                dy = cy - cy0
                
                # Normalize edilmiş sapmalar (Word dosyasında istenen e_y, e_x formatı)
                ex = dx / (w / 2)
                ey = dy / (h / 2)
                
                # Dead-band (Ölü bölge) Kontrolcüsü Kararı
                if abs(ex) < 0.1: # %10'dan az sapma varsa hizalı
                    command = "HIZALI"
                elif ex > 0:
                    command = "SAGA HIZALA"
                else:
                    command = "SOLA HIZALA"
                    
                cv2.putText(frame, f"Komut: {command} (ex: {ex:.2f})", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Ekranda göster
            cv2.imshow("Video Takip (Adim 4)", frame)
            
            # Loga yazdır
            writer.writerow([frame_count, timestamp, dx, dy, round(ex, 3), round(ey, 3), command])
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Log dosyası kaydedildi: {output_csv}")

if __name__ == "__main__":
    # Test için kendi kameranızı (0) kullanabilirsiniz.
    run_video_tracking(video_source=0)
