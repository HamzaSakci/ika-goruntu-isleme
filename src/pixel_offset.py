"""Piksel fark hesaplama: görüntü merkezine göre nesne sapması.

İşaret kuralı (OpenCV koordinatı):
    dx > 0 -> nesne merkezin SAĞINDA      dx < 0 -> SOLUNDA
    dy > 0 -> nesne merkezin ALTINDA      dy < 0 -> ÜSTÜNDE
"""
import math

DEADBAND = 0.10  # |ex| bu değerin altındaysa "HIZALI"


def image_center(w, h):
    """Görüntü merkezi (cx0, cy0). Tam sayı bölme: piksel indeksi tamsayıdır."""
    if w <= 0 or h <= 0:
        raise ValueError("Genişlik ve yükseklik pozitif olmalı")
    return w // 2, h // 2


def compute_offset(cx, cy, w, h):
    """
    Nesne merkezi (cx, cy) için merkeze göre sapmayı döndürür.
    Dönüş: dict(dx, dy, ex, ey, dist)
      ex, ey: [-1, 1] aralığında normalize sapma (çözünürlükten bağımsız)
    """
    cx0, cy0 = image_center(w, h)
    dx, dy = cx - cx0, cy - cy0
    return {
        "dx": dx,
        "dy": dy,
        "ex": dx / (w / 2),
        "ey": dy / (h / 2),
        "dist": math.hypot(dx, dy),
    }


def decide_command(ex, deadband=DEADBAND):
    """Ölü bölgeli (dead-band) oransal karar: HIZALI / SAGA HIZALA / SOLA HIZALA."""
    if abs(ex) < deadband:
        return "HIZALI"
    return "SAGA HIZALA" if ex > 0 else "SOLA HIZALA"
