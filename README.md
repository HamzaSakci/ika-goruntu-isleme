# İKA Görüntü İşleme — Piksel Fark Hesaplama

İKA Otonomi & Görüntü İşleme 2027 Yetkinlik Programı — **Level 1** çalışması.

## Amaç
1. Ana görüntünün merkezini bulmak
2. Görüntüde nesne tespiti (duba, levha, hedef)
3. Tespit edilen nesnenin görüntü merkezine **X/Y piksel uzaklığını** hesaplamak (`dx`, `dy`, normalize `ex`, `ey`)

## İşaret Kuralı
- `dx = cx − W/2` → **+** ise nesne merkezin **sağında**
- `dy = cy − H/2` → **+** ise nesne merkezin **altında** (OpenCV koordinatı)
- `ex = dx / (W/2)`, `ey = dy / (H/2)` → [-1, 1] aralığında, çözünürlükten bağımsız

## Kurulum
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Klasör Yapısı
```
src/        # merkez, tespit, piksel fark, video çalıştırıcı
tools/      # HSV ayarlayıcı vb. yardımcı scriptler
tests/      # pytest birim testleri
data/       # küçük örnek görüntüler (veri setleri repoda tutulmaz)
outputs/    # görsel çıktılar
docs/       # referans analizi, teknik not
```

## Veri Setleri
Roboflow kaynaklı (duba / levha / hedef). Boyut nedeniyle repoya eklenmez — linkler ve istatistikler buraya eklenecek.

## İlerleme
- [ ] Adım 1 — Görüntü merkezi
- [ ] Adım 2 — Nesne tespiti (HSV + contour, YOLO)
- [ ] Adım 3 — Piksel fark hesaplama
- [ ] Adım 4 — Video / webcam + CSV log
- [ ] Adım 5 — Birim testler
- [ ] Adım 6 — Dokümantasyon

## Referans
- [Poyqraz/Line-Tracking-and-Anomaly-Detection](https://github.com/Poyqraz/Line-Tracking-and-Anomaly-Detection) — Teknofest 2024 AUV kablo takibi
