"""
measurement_engine.py
----------------------
Core AI body-measurement engine for the AI Body Measurement Platform demo.

Pipeline (maps to project Modules 6, 7, 9, 10, 12):
    1. Human detection + 33-point pose landmarks   (MediaPipe Pose)
    2. Body segmentation / silhouette              (MediaPipe Selfie Segmentation)
    3. Scale calibration from known height          (pixels -> centimetres)
    4. Linear measurements from landmarks           (shoulder, arm, inseam, ...)
    5. Circumference estimation from silhouette     (chest / waist / hip girth)
    6. Per-measurement confidence scoring           (visibility + pose frontality)

The geometry functions are written so they can be unit-tested WITHOUT a real
image (see test_measurements.py) — the heavy CV models are only touched inside
`extract_from_image`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# MediaPipe's 33-landmark Pose model indices we care about.
L = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
}


@dataclass
class Measurement:
    """A single body measurement with its quality metadata (Module 12)."""

    name: str
    value_cm: float
    kind: str                      # "length" or "circumference"
    confidence: float              # 0..1
    method: str                    # how it was derived (for the report)

    @property
    def value_inch(self) -> float:
        return self.value_cm / 2.54

    @property
    def error_margin_cm(self) -> float:
        """Rough +/- error band that grows as confidence drops."""
        base = 0.8 if self.kind == "length" else 2.0
        return round(base + (1.0 - self.confidence) * base * 3, 1)


@dataclass
class ScanResult:
    measurements: dict[str, Measurement] = field(default_factory=dict)
    overall_confidence: float = 0.0
    person_detected: bool = False
    notes: list[str] = field(default_factory=list)

    def add(self, m: Measurement) -> None:
        self.measurements[m.name] = m

    def get_cm(self, name: str) -> Optional[float]:
        m = self.measurements.get(name)
        return m.value_cm if m else None


# --------------------------------------------------------------------------- #
# Pure geometry helpers (no CV dependency -> fully unit-testable)
# --------------------------------------------------------------------------- #
def _px(landmarks: np.ndarray, idx: int) -> np.ndarray:
    """Return the (x, y) pixel coordinate of a landmark row."""
    return landmarks[idx, :2]


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def polyline_length(points: list[np.ndarray]) -> float:
    return sum(euclidean(points[i], points[i + 1]) for i in range(len(points) - 1))


def pixel_height(landmarks: np.ndarray) -> float:
    """
    Full standing pixel height: from the top of the head (estimated a little
    above the eyes) down to the lowest heel.
    """
    eye = (_px(landmarks, L["left_eye"]) + _px(landmarks, L["right_eye"])) / 2
    nose = _px(landmarks, L["nose"])
    # Head top sits ~ the eye->nose vertical gap above the eyes.
    head_top_y = eye[1] - abs(nose[1] - eye[1]) * 1.6
    heel_y = max(_px(landmarks, L["left_heel"])[1], _px(landmarks, L["right_heel"])[1])
    return float(heel_y - head_top_y)


def cm_per_pixel(landmarks: np.ndarray, known_height_cm: float) -> float:
    """Scale calibration (Module 9 — perspective/scale)."""
    ph = pixel_height(landmarks)
    if ph <= 0:
        raise ValueError("Could not establish pixel height for scale.")
    return known_height_cm / ph


def frontality(landmarks: np.ndarray) -> float:
    """
    0..1 score of how square-to-camera the subject is. A perfectly frontal
    pose has symmetric shoulder/hip widths; a turned torso shrinks one side.
    Used to down-weight confidence (Module 12).
    """
    sh_w = euclidean(_px(landmarks, L["left_shoulder"]), _px(landmarks, L["right_shoulder"]))
    hip_w = euclidean(_px(landmarks, L["left_hip"]), _px(landmarks, L["right_hip"]))
    if sh_w <= 0:
        return 0.0
    ratio = hip_w / sh_w
    # Healthy frontal range for hip/shoulder ~ 0.55..0.95; penalise outside it.
    ideal = 0.75
    return float(max(0.0, 1.0 - min(1.0, abs(ratio - ideal) / 0.6)))


# --------------------------------------------------------------------------- #
# Landmark-based linear measurements (Module 10)
# --------------------------------------------------------------------------- #
def _visibility(landmarks: np.ndarray, idx: int) -> float:
    return float(landmarks[idx, 3]) if landmarks.shape[1] > 3 else 1.0


def _conf(landmarks: np.ndarray, idxs: list[int], front: float) -> float:
    vis = np.mean([_visibility(landmarks, i) for i in idxs])
    return round(float(vis) * (0.5 + 0.5 * front), 3)


def linear_measurements(landmarks: np.ndarray, scale: float, front: float) -> list[Measurement]:
    """Distances between landmarks, converted to cm via `scale` (cm/pixel)."""
    out: list[Measurement] = []

    def dist(a: str, b: str) -> float:
        return euclidean(_px(landmarks, L[a]), _px(landmarks, L[b])) * scale

    # Shoulder width (bony acromion-to-acromion; widened slightly for soft tissue)
    out.append(Measurement(
        "shoulder_width", dist("left_shoulder", "right_shoulder") * 1.05, "length",
        _conf(landmarks, [L["left_shoulder"], L["right_shoulder"]], front),
        "acromion-to-acromion landmark distance",
    ))

    # Arm length: shoulder -> elbow -> wrist (use the more visible arm)
    for side in ("left", "right"):
        pts = [_px(landmarks, L[f"{side}_shoulder"]),
               _px(landmarks, L[f"{side}_elbow"]),
               _px(landmarks, L[f"{side}_wrist"])]
        out.append(Measurement(
            f"arm_length_{side}", polyline_length(pts) * scale, "length",
            _conf(landmarks, [L[f"{side}_shoulder"], L[f"{side}_elbow"], L[f"{side}_wrist"]], front),
            "shoulder->elbow->wrist polyline",
        ))

    # Sleeve length ~ shoulder to wrist straight (garment sleeve)
    out.append(Measurement(
        "sleeve_length", ((dist("left_shoulder", "left_wrist") + dist("right_shoulder", "right_wrist")) / 2),
        "length",
        _conf(landmarks, [L["left_shoulder"], L["left_wrist"]], front),
        "shoulder-to-wrist average of both arms",
    ))

    # Inseam: hip -> knee -> ankle (inner leg length)
    for side in ("left", "right"):
        pts = [_px(landmarks, L[f"{side}_hip"]),
               _px(landmarks, L[f"{side}_knee"]),
               _px(landmarks, L[f"{side}_ankle"])]
        out.append(Measurement(
            f"inseam_{side}", polyline_length(pts) * scale, "length",
            _conf(landmarks, [L[f"{side}_hip"], L[f"{side}_knee"], L[f"{side}_ankle"]], front),
            "hip->knee->ankle polyline",
        ))

    # Outseam: hip to ankle (trouser outer length)
    out.append(Measurement(
        "outseam", ((dist("left_hip", "left_ankle") + dist("right_hip", "right_ankle")) / 2),
        "length",
        _conf(landmarks, [L["left_hip"], L["left_ankle"]], front),
        "hip-to-ankle average of both legs",
    ))

    # Torso length: shoulder mid to hip mid
    sh_mid = (_px(landmarks, L["left_shoulder"]) + _px(landmarks, L["right_shoulder"])) / 2
    hip_mid = (_px(landmarks, L["left_hip"]) + _px(landmarks, L["right_hip"])) / 2
    out.append(Measurement(
        "torso_length", euclidean(sh_mid, hip_mid) * scale, "length",
        _conf(landmarks, [L["left_shoulder"], L["left_hip"]], front),
        "shoulder-midpoint to hip-midpoint",
    ))

    return out


# --------------------------------------------------------------------------- #
# Silhouette-based circumferences (Module 10 — the hard part)
# --------------------------------------------------------------------------- #
def _row_width_px(mask: np.ndarray, y: int) -> float:
    """Width of the body silhouette (in px) at image row y."""
    if y < 0 or y >= mask.shape[0]:
        return 0.0
    row = np.where(mask[y] > 0)[0]
    return float(row.max() - row.min()) if row.size else 0.0


def _band_width_px(mask: np.ndarray, y: int, half: int = 3) -> float:
    """Median width over a small band of rows -> robust to noise."""
    ys = range(max(0, y - half), min(mask.shape[0], y + half + 1))
    widths = [w for w in (_row_width_px(mask, yy) for yy in ys) if w > 0]
    return float(np.median(widths)) if widths else 0.0


def ellipse_circumference(width_cm: float, depth_ratio: float) -> float:
    """
    Estimate a body-part girth by modelling the cross-section as an ellipse.
    Front-view gives the width (major axis 2a); depth (2b) is approximated as
    a fraction of the width. Ramanujan's approximation for the perimeter.
    """
    a = width_cm / 2.0
    b = a * depth_ratio
    h = ((a - b) ** 2) / ((a + b) ** 2) if (a + b) else 0
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


# Body-part vertical position (as a fraction between two anchor landmarks)
# and the depth:width ratio typical for that part (from anthropometric tables).
_CIRC_SPEC = {
    "chest":  {"anchor": "shoulder_hip", "t": 0.18, "depth": 0.72, "wfix": 1.02},
    "waist":  {"anchor": "shoulder_hip", "t": 0.75, "depth": 0.78, "wfix": 1.00},
    "hip":    {"anchor": "hip_knee",     "t": 0.10, "depth": 0.82, "wfix": 1.06},
    "thigh":  {"anchor": "hip_knee",     "t": 0.35, "depth": 0.90, "wfix": 0.62},
}


def circumference_measurements(landmarks: np.ndarray, mask: Optional[np.ndarray],
                               scale: float, front: float) -> list[Measurement]:
    if mask is None:
        return []
    out: list[Measurement] = []
    sh_mid = (_px(landmarks, L["left_shoulder"]) + _px(landmarks, L["right_shoulder"])) / 2
    hip_mid = (_px(landmarks, L["left_hip"]) + _px(landmarks, L["right_hip"])) / 2
    knee_mid = (_px(landmarks, L["left_knee"]) + _px(landmarks, L["right_knee"])) / 2

    anchors = {"shoulder_hip": (sh_mid, hip_mid), "hip_knee": (hip_mid, knee_mid)}

    for name, spec in _CIRC_SPEC.items():
        p0, p1 = anchors[spec["anchor"]]
        y = int(round(p0[1] + (p1[1] - p0[1]) * spec["t"]))
        width_px = _band_width_px(mask, y) * spec["wfix"]
        if width_px <= 0:
            continue
        girth = ellipse_circumference(width_px * scale, spec["depth"])
        # circumference estimates are inherently less certain -> cap confidence
        conf = round(min(0.85, (0.4 + 0.6 * front)) *
                     _visibility(landmarks, L["left_hip"]), 3)
        out.append(Measurement(
            name, girth, "circumference", conf,
            f"silhouette width @ t={spec['t']} + elliptical model (depth {spec['depth']})",
        ))
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_scan_result(landmarks: np.ndarray, mask: Optional[np.ndarray],
                      known_height_cm: float) -> ScanResult:
    """Assemble every measurement from landmarks (+ optional silhouette mask)."""
    res = ScanResult(person_detected=True)
    scale = cm_per_pixel(landmarks, known_height_cm)
    front = frontality(landmarks)

    res.add(Measurement("height", known_height_cm, "length", 1.0, "user-provided reference"))
    for m in linear_measurements(landmarks, scale, front):
        res.add(m)
    for m in circumference_measurements(landmarks, mask, scale, front):
        res.add(m)

    if front < 0.6:
        res.notes.append("Subject not fully frontal — retake for higher accuracy.")
    res.overall_confidence = round(
        float(np.mean([m.confidence for m in res.measurements.values()])), 3
    )
    return res


# --------------------------------------------------------------------------- #
# The one function that actually needs the CV models + an image
# --------------------------------------------------------------------------- #
def extract_from_image(image_bgr, known_height_cm: float):
    """
    Run the full CV pipeline on a single BGR image (OpenCV format).

    Returns (ScanResult, landmarks_or_None, mask_or_None).
    Imports MediaPipe lazily so the geometry can be tested without it.
    """
    import cv2
    import mediapipe as mp

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2,
                                enable_segmentation=True,
                                min_detection_confidence=0.5) as pose:
        result = pose.process(rgb)

    if not result.pose_landmarks:
        return ScanResult(person_detected=False,
                          notes=["No person detected — check framing and lighting."]), None, None

    lm = result.pose_landmarks.landmark
    landmarks = np.array([[p.x * w, p.y * h, p.z * w, p.visibility] for p in lm], dtype=float)

    mask = None
    if result.segmentation_mask is not None:
        mask = (result.segmentation_mask > 0.5).astype(np.uint8) * 255

    scan = build_scan_result(landmarks, mask, known_height_cm)
    return scan, landmarks, mask
