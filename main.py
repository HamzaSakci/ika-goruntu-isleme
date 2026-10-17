"""
IKA Görüntü İşleme - Level 1 Ana Çalıştırma Dosyası (Hepsi Bir Arada)
====================================================================
Bu dosya projenin tüm adımlarını tek bir yerden çalıştırmanı sağlar:
  1) Görüntü merkezini bulma (01_merkez.png)
  2) Tek bir resimde duba tespiti ve piksel fark (dx, dy) hesabı (02_tespit.png)
  3) 170 resmin tamamında doğruluk testi ve hata analizi (eval_results.csv, eval_worst.png)
  4) 22 adet birim testin (unit test) çalıştırılması
  5) Canlı kamera (webcam) ile takip ve CSV loglama
  6) HSV Renk Ayarlayıcı (Trackbar aracı)

Kullanım:
  python main.py          -> İnteraktif menüyü açar
  python main.py --hepsi  -> 1, 2, 3 ve 4. adımları sırayla otomatik çalıştırır
"""

import os
import sys
import glob
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

sys.path.insert(0, SRC_DIR)

from center import find_image_center  # noqa: E402
from detector import process_duba_detection  # noqa: E402
from pixel_offset import compute_offset, decide_command  # noqa: E402
from run_video import run_video_tracking  # noqa: E402


def adim1_goruntu_merkezi():
    print("\n--- [ADIM 1] Görüntü Merkezi Bulma ---")
    import cv2
    import numpy as np

    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    merkez = find_image_center(test_img)
    print(f"640x480 görüntünün hesaplanan merkezi: {merkez}")

    cv2.drawMarker(test_img, merkez, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, "01_merkez.png")

    ok, buf = cv2.imencode(".png", test_img)
    if ok:
        buf.tofile(out_path)
        print(f"Kaydedildi -> {out_path}")


def adim2_ve_3_tek_resim_tespit():
    print("\n--- [ADIM 2 & 3] Duba Tespiti ve Piksel Fark (dx, dy) Hesabı ---")
    test_dir = os.path.join(ROOT_DIR, "data", "duba_dataset", "test")
    images = glob.glob(os.path.join(test_dir, "images", "*.jpg"))
    if not images:
        print("Hata: data/duba_dataset/test/images içinde resim bulunamadı!")
        return

    secilen_resim = images[0]
    isim = os.path.splitext(os.path.basename(secilen_resim))[0]
    etiket_yolu = os.path.join(test_dir, "labels", isim + ".txt")
    cikti_yolu = os.path.join(OUTPUTS_DIR, "02_tespit.png")

    print(f"İşlenen resim: {os.path.basename(secilen_resim)}")
    process_duba_detection(secilen_resim, cikti_yolu, etiket_yolu)

    # Örnek bir normalize sapma ve komut hesabını da ekranda gösterelim
    ornek = compute_offset(475, 491, 640, 640)
    komut = decide_command(ornek["ex"])
    print(
        f"Normalize sapma: ex={ornek['ex']:.3f}, ey={ornek['ey']:.3f} "
        f"| Öklid mesafesi={ornek['dist']:.1f}px | Karar: {komut}"
    )


def adim_toplu_degerlendirme():
    print("\n--- [TOPLU ÖLÇÜM] 170 Görsel Üzerinde Değerlendirme ---")
    eval_script = os.path.join(TOOLS_DIR, "evaluate.py")
    subprocess.run([sys.executable, eval_script], cwd=ROOT_DIR, check=False)


def adim5_birim_testler():
    print("\n--- [ADIM 5] Birim Testler (Pytest - 22 Test) ---")
    subprocess.run([sys.executable, "-m", "pytest", "tests", "-v"], cwd=ROOT_DIR, check=False)


def adim4_canli_kamera():
    print("\n--- [ADIM 4] Canlı Kamera (Webcam) Takibi ve CSV Loglama ---")
    print("Kameradan çıkmak için açılan pencereye tıklayıp 'q' tuşuna bas.")
    csv_path = os.path.join(OUTPUTS_DIR, "tracking_log.csv")
    run_video_tracking(video_source=0, output_csv=csv_path)


def arac_hsv_ayarlayici():
    print("\n--- [ARAÇ] HSV Renk Ayarlayıcı (Trackbar) ---")
    tuner_script = os.path.join(TOOLS_DIR, "hsv_tuner.py")
    subprocess.run([sys.executable, tuner_script], cwd=ROOT_DIR, check=False)


def hepsini_calistir():
    adim1_goruntu_merkezi()
    adim2_ve_3_tek_resim_tespit()
    adim_toplu_degerlendirme()
    adim5_birim_testler()
    print("\n========================================================")
    print("Tüm otomatik adımlar tamamlandı! Çıktılar 'outputs/' klasöründe.")
    print("========================================================")


def menu():
    while True:
        print("\n" + "=" * 56)
        print("  İKA GÖRÜNTÜ İŞLEME - LEVEL 1 ANA MENÜ")
        print("=" * 56)
        print("  1) Adım 1: Görüntü merkezini bul (01_merkez.png)")
        print("  2) Adım 2-3: Duba tespiti ve dx/dy fark hesabı (02_tespit.png)")
        print("  3) Toplu Ölçüm: 170 resimde doğruluk ve hata analizi")
        print("  4) Adım 5: Birim testleri çalıştır (22 test)")
        print("  5) Adım 4: Canlı kamera (Webcam) takibi ve CSV log")
        print("  6) Araç: HSV Renk Ayarlayıcı (Trackbar)")
        print("  7) Otomatik: 1 + 2 + 3 + 4. adımların hepsini çalıştır")
        print("  0) Çıkış")
        secim = input("\nSeçimin (0-7): ").strip()

        if secim == "1":
            adim1_goruntu_merkezi()
        elif secim == "2":
            adim2_ve_3_tek_resim_tespit()
        elif secim == "3":
            adim_toplu_degerlendirme()
        elif secim == "4":
            adim5_birim_testler()
        elif secim == "5":
            adim4_canli_kamera()
        elif secim == "6":
            arac_hsv_ayarlayici()
        elif secim == "7":
            hepsini_calistir()
        elif secim == "0":
            print("Çıkılıyor...")
            break
        else:
            print("Geçersiz seçim, lütfen 0-7 arası bir numara gir.")


if __name__ == "__main__":
    if "--hepsi" in sys.argv:
        hepsini_calistir()
    else:
        menu()
