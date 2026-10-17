"""
Toplu değerlendirme: HSV duba tespitini etiketlerle karşılaştırır.

Çıktılar (outputs/):
  - eval_results.csv   : görsel başına sonuç
  - eval_hist.png      : piksel hatası histogramı
  - eval_worst.png     : en kötü 6 örnek (yeşil=tespit, kırmızı=etiket)
Konsola özet metrikler yazılır.
"""
import os
import sys
import csv
import glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__) + "/../src")
from detector import detect_duba, read_image_utf8  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "duba_dataset")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)


def gt_boxes(label_path, w, h):
    """Etiketteki TÜM dubaların kutularını (x1,y1,x2,y2) piksel olarak döndürür."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    for line in open(label_path):
        p = line.split()
        v = [float(x) for x in p[1:]]
        if len(v) == 4:
            xc, yc, bw, bh = v
            boxes.append(((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
        elif len(v) >= 6:
            xs, ys = v[0::2], v[1::2]
            boxes.append((min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h))
    return boxes


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def save_png(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(path)


rows = []
for split in ("train", "valid", "test"):
    for img_path in sorted(glob.glob(os.path.join(DATA, split, "images", "*.*"))):
        img = read_image_utf8(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        name = os.path.splitext(os.path.basename(img_path))[0]
        gts = gt_boxes(os.path.join(DATA, split, "labels", name + ".txt"), w, h)
        cx, cy, box, _ = detect_duba(img)
        if not gts:
            continue
        if cx is None:
            rows.append(dict(split=split, image=name, found=0, err_px=None, err_norm=None, iou=0.0,
                             det=None, gts=gts, path=img_path, w=w, h=h))
            continue
        x, y, bw, bh = box
        det = (x, y, x + bw, y + bh)
        # en yakın etiket merkezi
        best = min(gts, key=lambda g: np.hypot(cx - (g[0] + g[2]) / 2, cy - (g[1] + g[3]) / 2))
        gx, gy = (best[0] + best[2]) / 2, (best[1] + best[3]) / 2
        err = float(np.hypot(cx - gx, cy - gy))
        diag = float(np.hypot(w, h))
        rows.append(dict(split=split, image=name, found=1, err_px=err, err_norm=err / diag,
                         iou=iou(det, best), det=det, gts=gts, path=img_path, w=w, h=h))

# --- CSV ---
with open(os.path.join(OUT, "eval_results.csv"), "w", newline="", encoding="utf-8") as f:
    wr = csv.writer(f)
    wr.writerow(["split", "image", "found", "err_px", "err_norm_diag", "iou"])
    for r in rows:
        wr.writerow([r["split"], r["image"], r["found"],
                     "" if r["err_px"] is None else round(r["err_px"], 1),
                     "" if r["err_norm"] is None else round(r["err_norm"], 4), round(r["iou"], 3)])

# --- Özet ---
n = len(rows)
found = [r for r in rows if r["found"]]
errs = np.array([r["err_norm"] for r in found])
ious = np.array([r["iou"] for r in found])
print(f"Toplam görsel (etiketli): {n}")
print(f"Duba bulunan: {len(found)} ({100*len(found)/n:.1f}%) | Bulunamayan (miss): {n-len(found)} ({100*(n-len(found))/n:.1f}%)")
print(f"Merkez hatası (köşegene oranla)  ortanca: {np.median(errs)*100:.1f}%  ort: {errs.mean()*100:.1f}%")
print(f"IoU  ortanca: {np.median(ious):.2f}  ort: {ious.mean():.2f}")
for thr in (0.05, 0.10):
    ok = int((errs < thr).sum())
    print(f"Hata < köşegenin %{int(thr*100)}'i olan: {ok}/{n} ({100*ok/n:.1f}%)")
print("Split bazında bulma oranı:", {s: f"{sum(r['found'] for r in rows if r['split']==s)}/{sum(1 for r in rows if r['split']==s)}" for s in ('train', 'valid', 'test')})

# --- Histogram (matplotlib) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 4))
plt.hist(errs * 100, bins=25, color="#e8743b")
plt.axvline(np.median(errs) * 100, color="k", ls="--", label=f"ortanca %{np.median(errs)*100:.1f}")
plt.xlabel("Merkez hatası (% köşegen)")
plt.ylabel("Görsel sayısı")
plt.title("HSV duba tespiti - merkez hatası dağılımı")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "eval_hist.png"), dpi=120)
plt.close()

# --- En iyi 6 ve En kötü 6 ---
def make_montage(subset, out_name):
    tiles = []
    for r in subset[:6]:
        im = read_image_utf8(r["path"])
        h_im, w_im = im.shape[:2]
        cx0, cy0 = w_im // 2, h_im // 2
        cv2.drawMarker(im, (cx0, cy0), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)
        for g in r["gts"]:
            cv2.rectangle(im, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 0, 255), 3)
        if r["det"]:
            d = r["det"]
            cv2.rectangle(im, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 0), 3)
            cx_d, cy_d = (int(d[0]) + int(d[2])) // 2, (int(d[1]) + int(d[3])) // 2
            cv2.circle(im, (cx_d, cy_d), 6, (0, 255, 0), -1)
            cv2.line(im, (cx0, cy0), (cx_d, cy_d), (0, 255, 255), 2)
        label = "MISS" if not r["found"] else f"hata %{r['err_norm']*100:.1f} ({r['err_px']:.0f}px)"
        cv2.putText(im, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        tiles.append(cv2.resize(im, (360, 360)))
    while len(tiles) < 6:
        tiles.append(np.zeros((360, 360, 3), np.uint8))
    montage = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
    save_png(os.path.join(OUT, out_name), montage)


worst = sorted(rows, key=lambda r: -(1.0 if not r["found"] else r["err_norm"]))[:6]
best = sorted([r for r in rows if r["found"]], key=lambda r: r["err_norm"])[:6]
make_montage(worst, "eval_worst.png")
make_montage(best, "eval_best.png")
print("Çıktılar: outputs/eval_results.csv, eval_hist.png, eval_best.png, eval_worst.png")
