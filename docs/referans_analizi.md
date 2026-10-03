# Referans Repo Analizi — Poyqraz/Line-Tracking-and-Anomaly-Detection

> Bu dosya kendi yorumlarımla doldurulacak.

## Temel mantık
```
frame → HSV → inRange → morfoloji → findContours → en büyük contour
      → moments ile merkez (cx, cy) → ekran merkezi (W/2, H/2)
      → dx = cx - W/2, dy = cy - H/2 → dead-band (40 px) → "Hizalı" / "Sağa-Sola hizala"
```

## Tespit edilen sorunlar
| Satır | Sorun |
|---|---|
| 206–209 | OPEN sonucu CLOSE'da eziliyor; findContours ham maskeye uygulanıyor |
| 240–249 | screen_center yoksa dx, dy tanımsız (UnboundLocalError) |
| 72–73 | preprocess_image erken return — CLAHE ölü kod |
| 493–503 | Işığa göre hesaplanan eşikler kullanılmıyor |
| 228 | Dikey çizgide vx = 0 → sıfıra bölme |
| 390–400 | predict_turn her zaman None döndürüyor |
| 516 | 640/480 sabit kodlanmış |
| 611–614 | Videoya farklı boyutta iki kare yazılıyor |
| genel | Piksel cinsinden eşikler, birim test yok |

## İKA'ya uyarlama
- `dx` → yaw / direksiyon hatası
- `dy` → kamera sabit açılı olduğundan nesne uzaklığıyla ilişkili (ileride homografi)

## Kendi yorumlarım
- ...
