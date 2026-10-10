import numpy as np
import cv2
import pytest

from center import find_image_center
from pixel_offset import image_center, compute_offset, decide_command
from detector import detect_duba, load_yolo_center


def orange_scene(w=640, h=480, cx=320, cy=240, half=30):
    """Siyah zemin üzerine bilinen konumda turuncu kare çizer (BGR turuncu ~ HSV H=10)."""
    img = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (0, 128, 255), -1)
    return img


# ---- Adım 1: merkez ----
def test_center_even_size():
    assert find_image_center(np.zeros((480, 640, 3), np.uint8)) == (320, 240)


def test_center_odd_size_uses_floor_division():
    assert find_image_center(np.zeros((481, 641, 3), np.uint8)) == (320, 240)


def test_center_grayscale_image():
    assert find_image_center(np.zeros((100, 200), np.uint8)) == (100, 50)


def test_image_center_invalid_size_raises():
    with pytest.raises(ValueError):
        image_center(0, 480)


# ---- Adım 3: piksel fark ----
def test_offset_object_at_center_is_zero():
    r = compute_offset(320, 240, 640, 480)
    assert (r["dx"], r["dy"], r["ex"], r["ey"], r["dist"]) == (0, 0, 0.0, 0.0, 0.0)


def test_offset_right_and_below_are_positive():
    r = compute_offset(420, 290, 640, 480)
    assert r["dx"] == 100 and r["dy"] == 50


def test_offset_left_and_above_are_negative():
    r = compute_offset(220, 190, 640, 480)
    assert r["dx"] == -100 and r["dy"] == -50


def test_offset_normalized_edges_are_plus_minus_one():
    assert compute_offset(640, 240, 640, 480)["ex"] == pytest.approx(1.0)
    assert compute_offset(0, 240, 640, 480)["ex"] == pytest.approx(-1.0)
    assert compute_offset(320, 480, 640, 480)["ey"] == pytest.approx(1.0)


def test_offset_normalization_is_resolution_independent():
    a = compute_offset(480, 240, 640, 480)["ex"]
    b = compute_offset(960, 360, 1280, 720)["ex"]
    assert a == pytest.approx(b)


def test_offset_distance_is_euclidean():
    assert compute_offset(323, 244, 640, 480)["dist"] == pytest.approx(5.0)


# ---- Komut kararı (dead-band) ----
def test_command_deadband_aligned():
    assert decide_command(0.05) == "HIZALI"
    assert decide_command(-0.05) == "HIZALI"


def test_command_right_and_left():
    assert decide_command(0.5) == "SAGA HIZALA"
    assert decide_command(-0.5) == "SOLA HIZALA"


def test_command_deadband_boundary_is_not_aligned():
    assert decide_command(0.10) == "SAGA HIZALA"


# ---- Adım 2: tespit (sentetik görüntü) ----
def test_detect_finds_object_at_known_position():
    cx, cy, box, _ = detect_duba(orange_scene(cx=400, cy=300))
    assert abs(cx - 400) <= 2 and abs(cy - 300) <= 2
    assert box is not None


def test_detect_matches_expected_offset():
    img = orange_scene(cx=420, cy=290)
    cx, cy, _, _ = detect_duba(img)
    r = compute_offset(cx, cy, 640, 480)
    assert abs(r["dx"] - 100) <= 2 and abs(r["dy"] - 50) <= 2


def test_detect_no_object_returns_none():
    cx, cy, box, _ = detect_duba(np.zeros((480, 640, 3), np.uint8))
    assert cx is None and cy is None and box is None


def test_detect_ignores_tiny_noise():
    img = np.zeros((480, 640, 3), np.uint8)
    img[100:103, 100:103] = (0, 128, 255)
    assert detect_duba(img)[0] is None


def test_detect_picks_largest_of_two_objects():
    img = orange_scene(cx=150, cy=240, half=15)
    cv2.rectangle(img, (470, 190), (570, 290), (0, 128, 255), -1)
    cx, cy, _, _ = detect_duba(img)
    assert abs(cx - 520) <= 2 and abs(cy - 240) <= 2


def test_detect_ignores_non_orange_color():
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.rectangle(img, (200, 200), (300, 300), (255, 0, 0), -1)  # mavi
    assert detect_duba(img)[0] is None


# ---- Etiket okuma ----
def test_label_bbox_format(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.4\n")
    cx, cy, box = load_yolo_center(str(p), 640, 480)
    assert (cx, cy) == (320, 240)
    assert box == (256, 144, 384, 336)


def test_label_polygon_format(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
    cx, cy, box = load_yolo_center(str(p), 100, 100)
    assert (cx, cy) == (50, 50)
    assert box == (25, 25, 75, 75)


def test_label_missing_file_returns_none(tmp_path):
    assert load_yolo_center(str(tmp_path / "yok.txt"), 640, 480) is None
