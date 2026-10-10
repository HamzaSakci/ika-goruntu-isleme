# Teknik Not — Piksel Fark Hesaplama (Level 1)

**Hazırlayan:** Hamza Sakçı  **Tarih:** 10 Ekim 2026
**Repo:** https://github.com/HamzaSakci/ika-goruntu-isleme

## 1. Amaç
İKA Otonomi 2027 programı Level 1 görevi: (1) görüntü merkezini bul, (2) görüntüde nesneyi tespit et (duba), (3) nesne merkezinin görüntü merkezine X/Y piksel uzaklığını hesapla. Bu, hedefe göre hata üreten tüm kontrol zincirinin temelidir (`e = nesne − merkez`).

## 2. Yöntem

### 2.1 Görüntü merkezi
`cx0 = W // 2`, `cy0 = H // 2` (`src/center.py`, `src/pixel_offset.py`). Piksel indeksi tamsayı olduğundan tam sayı bölme kullanıldı. 641 piksel genişlikte gerçek merkez 320,5'tir, indeks olarak 320 alınır (±0,5 px).

### 2.2 Nesne tespiti (klasik görüntü işleme, `src/detector.py`)
1. BGR → HSV dönüşümü (ışık değişimine RGB'den dayanıklı).
2. Turuncu/kırmızı için **iki HSV aralığı**: `H 0–25` ve `H 170–179`, `S ≥ 150`, `V ≥ 30`. Kırmızı H ekseninde 0'da başa sardığı ve dubanın gölgedeki kısmı H≈2–4'e kaydığı için tek aralık yetmedi.
3. Morfoloji: `OPEN` (5×5) ile gürültü silme, ardından **dikey uzun kernel ile `CLOSE`**. Dubanın beyaz şeritleri turuncu maskeyi parçalar; bu adım parçaları birleştirir.
4. `findContours` → en büyük kontur → `boundingRect`. Alan < 200 px² ise reddedilir.
5. Nesne merkezi = kutu merkezi.

### 2.3 Piksel fark (`src/pixel_offset.py`)
```
dx = cx − cx0     dy = cy − cy0     dist = √(dx² + dy²)
ex = dx / (W/2)   ey = dy / (H/2)   ∈ [−1, 1]
```
**İşaret kuralı (OpenCV):** `dx>0` nesne sağda, `dy>0` nesne altta. Normalize `ex/ey` çözünürlükten bağımsızdır. Komut: ölü bölgeli karar, `|ex| < 0.10` → HIZALI, aksi halde SAĞA/SOLA HİZALA.

### 2.4 Video ve loglama (`src/run_video.py`)
Webcam/videoda her kare için tespit + sapma + komut hesaplanır; `outputs/tracking_log.csv` dosyasına `Frame, Timestamp, dx, dy, ex, ey, Command` yazılır.

## 3. Veri
Roboflow `Denemeler_Montagne v6` (16 sınıf). `tools/filter_duba.py` ile yalnız `Duba` (sınıf 12) alındı: **170 görsel** (train 115 / valid 38 / test 17). Etiketler bbox değil **poligon** formatında; kutu, poligon köşelerinin min/max değerinden hesaplandı.

## 4. Sonuçlar (`tools/evaluate.py`)
Her görselde HSV tespiti etiketteki gerçek duba kutusuyla karşılaştırıldı. Merkez hatası görüntü köşegenine oranlanarak çözünürlükten bağımsız ölçüldü.

| Metrik | Değer |
|---|---|
| Merkez hatası, ortanca | **%1,8** |
| Merkez hatası, ortalama | %6,9 |
| Hata < köşegenin %5'i | 107/170 (%62,9) |
| Hata < köşegenin %10'u | 130/170 (%76,5) |
| IoU ortanca / ortalama | 0,45 / 0,46 |
| Birim test | 22/22 geçti |

![hist](../outputs/eval_hist.png)

## 5. Hata analizi (kök neden)
![worst](../outputs/eval_worst.png)

| Gözlem | Kök neden | Önlem |
|---|---|---|
| Kırmızı araba, turuncu bariyer/kamyon, sonbahar yaprakları dubayla karışıyor (yeşil kutu yanlış yerde) | Yalnız renk eşiği kullanılıyor; bu nesneler aynı HSV aralığında | Şekil (en-boy oranı), beyaz şerit deseni doğrulaması veya YOLO |
| Hiçbir görselde "duba yok" denmiyor (miss = 0) | Her karede bir kontur seçiliyor; güven eşiği yok | Alan/şekil/doygunluk tabanlı güven skoru |
| IoU düşük (0,45) | Gölgedeki kenar maskeden düşüyor, kutu daralıyor | Gölge için `V` eşiği/adaptif eşik, `CLAHE` |
| Çok dubalı karelerde tek kutu | Yalnız en büyük kontur seçiliyor | Birden fazla konturu ayrı ayrı raporlama |

**Ön düzeltmeler (geliştirme sırası):**
1. İlk sürümde yalnız en büyük turuncu parça alınıyordu → dubanın alt şeridi seçilip merkez kaydı (hata 133 px). Dikey `CLOSE` ve kutu merkezi ile 26 px'e indi.
2. `H` alt sınırı 5 iken gölgedeki kısım kayboluyordu → 0'a çekildi, kırmızı sarma eklendi. Bu, zemindeki turuncu çizgileri de maskeye aldı → `S` eşiği 100 → 150.
3. `cv2.imread/imwrite` Türkçe karakterli (`Masaüstü`) yolda sessizce başarısız oldu → `imdecode/imencode` + `numpy` ile okuma/yazma.

## 6. Sınırlar
- Renk tabanlı yöntem ışığa, arka plan rengine ve benzer renkli nesnelere duyarlı.
- `dy` yalnız piksel; metre için kamera kalibrasyonu/homografi gerekir (Level 1, hafta 5–6).
- Video testi gerçek kamerayla henüz yapılmadı.

## 7. Sonraki adımlar
1. Yanlış pozitif azaltma (şekil + şerit doğrulaması), aynı metrikle önce/sonra karşılaştırma.
2. YOLOv8n ile aynı veri üzerinde eğitim (Level 2) ve HSV ile tablo karşılaştırması.
3. Planlama: grid üzerinde BFS → Dijkstra → A\*.
