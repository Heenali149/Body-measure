# AI Body Measurement - Working Demo

A runnable prototype of the core AI engine behind the **AI Body Measurement Platform**:
a person is photographed once (full-body, frontal), and the engine returns tailoring
measurements, confidence scores, and recommended uniform sizes.

This is the hard, defensible part of the whole product - the computer-vision pipeline.
Everything else (auth, org management, dashboards, production workflow) is standard app
work; measurement accuracy is what makes the platform worth building.

## What it does

1. **Detects the person and 33 body landmarks** — MediaPipe Pose.
2. **Segments the body silhouette** — MediaPipe Selfie Segmentation.
3. **Calibrates scale** from the subject's known height (pixels → centimetres).
4. **Computes linear measurements** from landmark geometry: shoulder width, arm/sleeve
   length, inseam, outseam, torso length.
5. **Estimates circumferences** (chest, waist, hip, thigh) from silhouette width using an
   elliptical cross-section model — the standard monocular approach when there's no LiDAR.
6. **Scores every measurement** with a confidence value and an error margin derived from
   landmark visibility and how frontal the pose is.
7. **Recommends sizes** (shirt / jacket / blazer / trouser) with US/UK/EU conversion,
   slim/regular/loose fit ease, and a body-shape classification.

## Run it

```bash
pip install -r requirements.txt

# 1) No photo needed — runs the whole pipeline on synthetic data:
python scan.py --selftest

# 2) On a real full-body frontal photo (height in cm is the scale reference):
python scan.py --image person.jpg --height 178 --fit regular
```

Real-image runs also write `person_annotated.png` (skeleton + measurement overlay) and
`person_report.json` (machine-readable output for the backend/API).

For the best measurement accuracy: tight-fitting clothes, plain background, full body in
frame including feet, camera at roughly hip height, ~2.5–3 m away.

## Verify the math

```bash
python test_measurements.py            # 11 unit tests, no model/image required
# or: python -m pytest test_measurements.py -v
```

The tests feed synthetic landmarks with known true dimensions and assert the engine
recovers them (scale calibration, distances, ellipse perimeter, size bands, etc.).

## Files

| File | Purpose |
|------|---------|
| `measurement_engine.py` | CV pipeline + all measurement geometry and confidence scoring |
| `size_engine.py` | Size charts, regional conversion, fit ease, body-shape classifier |
| `scan.py` | CLI runner, printed report, annotated overlay, JSON export, self-test |
| `test_measurements.py` | Unit tests for the geometry/measurement math |
| `requirements.txt` | Dependencies |

## How this maps to the platform's 25 modules

| Module (spec) | Where it lives in this demo |
|---|---|
| 6 - Live AI Vision Engine (human/landmark/segmentation) | `extract_from_image` (MediaPipe Pose + Segmentation) |
| 7 - Auto Quality Validation | `frontality`, visibility-weighted confidence, "retake" notes |
| 9 - AI Processing Engine (scale, alignment, prediction) | `cm_per_pixel`, `linear_measurements`, `circumference_measurements` |
| 10 - Automatic Body Measurements | full measurement set (upper & lower body) |
| 11 - Body Shape Intelligence | `classify_body_shape` |
| 12 - AI Confidence Score | `Measurement.confidence` + `error_margin_cm` |
| 14 - Size Recommendation Engine | `size_engine.recommend` with US/UK/EU conversion |
| 17 - Digital Measurement Report | printed report + `*_report.json` + annotated image |

## Accuracy roadmap (what a production build adds)

This prototype is a single-view, geometry-first estimator - deliberately transparent and
testable. The path to production accuracy:

- **Multi-angle capture** (front + both sides) to measure true depth instead of estimating
  it, tightening every circumference.
- **A learned correction model** (regression / small net) trained on paired scan-vs-tape
  ground truth, replacing the fixed anthropometric ratios.
- **3D body reconstruction** (SMPL / SMPL-X parametric body models) — this is what unlocks
  the platform's "future" features too: 3D avatar, virtual try-on, and fabric-consumption
  estimation all fall out of a fitted body mesh.
- **On-device inference** via ONNX Runtime / TensorFlow Lite for the mobile app.

Reference systems solving the same problem commercially: Bodygram, 3DLook, Size Stream,
Nettelo - useful accuracy benchmarks to design against.
