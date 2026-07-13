"""
test_measurements.py
--------------------
Unit tests for the geometry + measurement math. These run WITHOUT MediaPipe or
any image — they feed synthetic landmark coordinates whose true dimensions we
know, and assert the engine recovers them within tolerance.

Run:  python -m pytest test_measurements.py -v
   or  python test_measurements.py   (falls back to a plain runner)
"""

import math

import numpy as np

import measurement_engine as me
import size_engine as se


def make_landmarks(height_cm_pixels=(150, 1210)):
    """
    Build a controlled pose. Head-top y and heel y define pixel height so we can
    verify scale calibration exactly.
    """
    cx = 360.0
    pts = {
        "nose": (cx, 150), "left_eye": (cx - 15, 135), "right_eye": (cx + 15, 135),
        "left_shoulder": (cx - 100, 300), "right_shoulder": (cx + 100, 300),
        "left_elbow": (cx - 100, 470), "right_elbow": (cx + 100, 470),
        "left_wrist": (cx - 100, 640), "right_wrist": (cx + 100, 640),
        "left_hip": (cx - 70, 700), "right_hip": (cx + 70, 700),
        "left_knee": (cx - 70, 950), "right_knee": (cx + 70, 950),
        "left_ankle": (cx - 70, 1180), "right_ankle": (cx + 70, 1180),
        "left_heel": (cx - 70, 1210), "right_heel": (cx + 70, 1210),
    }
    arr = np.zeros((33, 4), dtype=float)
    for name, (x, y) in pts.items():
        arr[me.L[name]] = (x, y, 0.0, 0.99)
    return arr


def approx(a, b, tol):
    return abs(a - b) <= tol


def test_euclidean():
    assert approx(me.euclidean(np.array([0, 0]), np.array([3, 4])), 5.0, 1e-9)


def test_polyline_length():
    pts = [np.array([0, 0]), np.array([0, 3]), np.array([4, 3])]
    assert approx(me.polyline_length(pts), 7.0, 1e-9)


def test_pixel_height_and_scale():
    lm = make_landmarks()
    # head_top_y = 135 - (150-135)*1.6 = 135 - 24 = 111 ; heel = 1210 -> 1099 px
    ph = me.pixel_height(lm)
    assert approx(ph, 1099.0, 1.0), ph
    scale = me.cm_per_pixel(lm, 178.0)
    assert approx(scale, 178.0 / 1099.0, 1e-6), scale


def test_shoulder_width_scaled():
    lm = make_landmarks()
    scale = me.cm_per_pixel(lm, 178.0)
    front = me.frontality(lm)
    ms = {m.name: m for m in me.linear_measurements(lm, scale, front)}
    # shoulder pixel width = 200 px, *1.05 soft-tissue factor
    expected = 200 * scale * 1.05
    assert approx(ms["shoulder_width"].value_cm, expected, 0.01)
    # sanity: a 178 cm person should have shoulder width ~ 30-38 cm
    assert 28 <= ms["shoulder_width"].value_cm <= 42, ms["shoulder_width"].value_cm


def test_inseam_reasonable():
    lm = make_landmarks()
    scale = me.cm_per_pixel(lm, 178.0)
    front = me.frontality(lm)
    ms = {m.name: m for m in me.linear_measurements(lm, scale, front)}
    inseam = ms["inseam_left"].value_cm
    # hip(700)->knee(950)->ankle(1180) is ~480 px vertical => ~ 77 cm; plausible
    assert 65 <= inseam <= 90, inseam


def test_frontality_range():
    lm = make_landmarks()
    f = me.frontality(lm)
    assert 0.0 <= f <= 1.0
    # symmetric synthetic pose should be fairly frontal
    assert f > 0.5, f


def test_ellipse_circumference_circle():
    # depth_ratio = 1 -> ellipse becomes a circle: perimeter = pi * diameter
    c = me.ellipse_circumference(width_cm=20.0, depth_ratio=1.0)
    assert approx(c, math.pi * 20.0, 0.05), c


def test_confidence_and_error_margin():
    lm = make_landmarks()
    scan = me.build_scan_result(lm, None, 178.0)
    for m in scan.measurements.values():
        assert 0.0 <= m.confidence <= 1.0
        assert m.error_margin_cm > 0
    assert 0.0 <= scan.overall_confidence <= 1.0


def test_size_engine_bands():
    # chest 96 cm -> M ; chest 104 -> L ; waist 82 cm -> W32
    assert se._lookup_top(96)[0] == "M"
    assert se._lookup_top(104)[0] == "L"
    assert se._lookup_trouser(82)[0] == "32"


def test_recommend_end_to_end():
    lm = make_landmarks()
    # need a mask for circumferences -> reuse scan.py's synthetic mask shape
    mask = np.zeros((1280, 720), dtype=np.uint8)
    for y in range(300, 700):
        mask[y, 255:465] = 255      # torso ~210 px wide
    for y in range(700, 950):
        mask[y, 290:430] = 255
    scan = me.build_scan_result(lm, mask, 178.0)
    recs = se.recommend(scan, "regular")
    garments = {r.garment for r in recs}
    assert {"shirt", "trouser"}.issubset(garments)
    for r in recs:
        assert r.primary_size


def test_body_shape_classifier():
    assert se.classify_body_shape(100, 80, 100) in {
        "athletic / inverted-V", "regular", "rectangle / straight"
    }
    assert se.classify_body_shape(95, 85, 108) == "pear / triangle"


# --------------------------------------------------------------------------- #
def _run_plain():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa
            print(f"  ERROR {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_plain() else 1)
