"""
scan.py
-------
Command-line runner for the AI Body Measurement demo.

Usage:
    python scan.py --image person.jpg --height 178 --fit regular

    # or run the built-in self-check without any photo / model:
    python scan.py --selftest

Outputs:
    * a printed measurement report with confidence + error margins
    * <image>_annotated.png  — skeleton + measurement overlay (if --image given)
    * <image>_report.json    — machine-readable result (for the backend/API)
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import measurement_engine as me
import size_engine as se


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(scan, recs, fit):
    if not scan.person_detected:
        print("\n[!] No person detected.")
        for n in scan.notes:
            print("    -", n)
        return

    print("\n" + "=" * 64)
    print(" AI BODY MEASUREMENT REPORT")
    print("=" * 64)
    print(f" Overall confidence : {scan.overall_confidence * 100:5.1f}%")
    print(f" Fit preference     : {fit}")
    if scan.notes:
        for n in scan.notes:
            print(f" Note               : {n}")

    print("\n --- Measurements ---------------------------------------------")
    print(f" {'Measurement':<18}{'cm':>8}{'inch':>8}{'± cm':>7}{'conf':>7}")
    print(" " + "-" * 60)
    for m in scan.measurements.values():
        print(f" {m.name:<18}{m.value_cm:>8.1f}{m.value_inch:>8.1f}"
              f"{m.error_margin_cm:>7.1f}{m.confidence * 100:>6.0f}%")

    chest = scan.get_cm("chest") or 0
    waist = scan.get_cm("waist") or 0
    hip = scan.get_cm("hip") or 0
    shape = se.classify_body_shape(chest, waist, hip)
    print(f"\n Body shape         : {shape}")

    print("\n --- Recommended sizes ----------------------------------------")
    for r in recs:
        alt = f"  (alt: {r.alternative})" if r.alternative else ""
        region = "  ".join(f"{k}:{v}" for k, v in r.regional.items())
        print(f" {r.garment:<9} -> {r.primary_size:<5}{alt}")
        print(f"            {region}   [{r.confidence * 100:.0f}% conf]")
        print(f"            {r.rationale}")
    print("=" * 64 + "\n")


def to_json(scan, recs, shape):
    return {
        "person_detected": scan.person_detected,
        "overall_confidence": scan.overall_confidence,
        "body_shape": shape,
        "notes": scan.notes,
        "measurements": {
            m.name: {
                "cm": round(m.value_cm, 1),
                "inch": round(m.value_inch, 1),
                "error_margin_cm": m.error_margin_cm,
                "confidence": m.confidence,
                "kind": m.kind,
                "method": m.method,
            }
            for m in scan.measurements.values()
        },
        "size_recommendations": [
            {
                "garment": r.garment,
                "size": r.primary_size,
                "alternative": r.alternative,
                "regional": r.regional,
                "confidence": r.confidence,
                "rationale": r.rationale,
            }
            for r in recs
        ],
    }


# --------------------------------------------------------------------------- #
# Annotated overlay
# --------------------------------------------------------------------------- #
def draw_overlay(image_bgr, landmarks, scan, out_path):
    import cv2

    img = image_bgr.copy()
    Lm = me.L

    # skeleton edges
    edges = [
        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ]
    for a, b in edges:
        pa = tuple(landmarks[Lm[a], :2].astype(int))
        pb = tuple(landmarks[Lm[b], :2].astype(int))
        cv2.line(img, pa, pb, (0, 255, 0), 2)
    for name, idx in Lm.items():
        p = tuple(landmarks[idx, :2].astype(int))
        cv2.circle(img, p, 4, (0, 140, 255), -1)

    # a few labelled measurements
    def midpoint(a, b):
        return tuple(((landmarks[Lm[a], :2] + landmarks[Lm[b], :2]) / 2).astype(int))

    labels = [
        ("shoulder_width", midpoint("left_shoulder", "right_shoulder")),
        ("chest", midpoint("left_shoulder", "left_hip")),
        ("waist", midpoint("left_hip", "right_hip")),
        ("inseam_left", midpoint("left_hip", "left_knee")),
    ]
    for name, pos in labels:
        m = scan.measurements.get(name)
        if not m:
            continue
        txt = f"{name}: {m.value_cm:.0f}cm"
        cv2.putText(img, txt, (pos[0] + 6, pos[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (pos[0] + 6, pos[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, img)


# --------------------------------------------------------------------------- #
# Self-test: exercises the full pipeline with synthetic landmarks (no model)
# --------------------------------------------------------------------------- #
def synthetic_landmarks() -> np.ndarray:
    """A plausible frontal standing pose in a 720x1280 frame (x, y, z, vis)."""
    cx = 360.0
    pts = {
        "nose": (cx, 150), "left_eye": (cx - 15, 135), "right_eye": (cx + 15, 135),
        "left_shoulder": (cx - 95, 300), "right_shoulder": (cx + 95, 300),
        "left_elbow": (cx - 120, 470), "right_elbow": (cx + 120, 470),
        "left_wrist": (cx - 135, 640), "right_wrist": (cx + 135, 640),
        "left_hip": (cx - 70, 700), "right_hip": (cx + 70, 700),
        "left_knee": (cx - 72, 950), "right_knee": (cx + 72, 950),
        "left_ankle": (cx - 74, 1180), "right_ankle": (cx + 74, 1180),
        "left_heel": (cx - 74, 1210), "right_heel": (cx + 74, 1210),
    }
    arr = np.zeros((33, 4), dtype=float)
    for name, (x, y) in pts.items():
        arr[me.L[name]] = (x, y, 0.0, 0.99)
    return arr


def synthetic_mask(landmarks: np.ndarray) -> np.ndarray:
    """Build a rough silhouette mask consistent with the synthetic pose."""
    mask = np.zeros((1280, 720), dtype=np.uint8)
    # torso trapezoid + legs, widths chosen to yield realistic girths
    import numpy as _np
    def fill_band(y0, y1, half_w):
        for y in range(int(y0), int(y1)):
            cx = 360
            mask[y, int(cx - half_w):int(cx + half_w)] = 255
    fill_band(300, 700, 105)   # torso
    fill_band(700, 950, 60)    # thighs region (combined)
    fill_band(950, 1200, 40)   # lower legs
    return mask


def run_selftest():
    lm = synthetic_landmarks()
    mask = synthetic_mask(lm)
    scan = me.build_scan_result(lm, mask, known_height_cm=178.0)
    recs = se.recommend(scan, fit_preference="regular")
    shape = se.classify_body_shape(scan.get_cm("chest") or 0,
                                   scan.get_cm("waist") or 0,
                                   scan.get_cm("hip") or 0)
    print_report(scan, recs, "regular")
    print("Self-test completed: pipeline ran end-to-end on synthetic input.")
    return scan


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="AI Body Measurement demo")
    ap.add_argument("--image", help="path to a full-body frontal photo")
    ap.add_argument("--height", type=float, help="subject height in cm (scale reference)")
    ap.add_argument("--fit", default="regular", choices=["slim", "regular", "loose"])
    ap.add_argument("--selftest", action="store_true",
                    help="run the pipeline on synthetic data (no photo/model needed)")
    args = ap.parse_args()

    if args.selftest or not args.image:
        run_selftest()
        return

    if args.height is None:
        print("Error: --height (cm) is required for scale calibration.")
        sys.exit(1)

    import cv2
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not read image '{args.image}'.")
        sys.exit(1)

    scan, landmarks, mask = me.extract_from_image(img, args.height)
    recs = se.recommend(scan, fit_preference=args.fit)
    shape = se.classify_body_shape(scan.get_cm("chest") or 0,
                                   scan.get_cm("waist") or 0,
                                   scan.get_cm("hip") or 0)
    print_report(scan, recs, args.fit)

    if landmarks is not None:
        base = args.image.rsplit(".", 1)[0]
        draw_overlay(img, landmarks, scan, f"{base}_annotated.png")
        with open(f"{base}_report.json", "w") as f:
            json.dump(to_json(scan, recs, shape), f, indent=2)
        print(f"Saved: {base}_annotated.png  and  {base}_report.json")


if __name__ == "__main__":
    main()
