"""
size_engine.py
--------------
Size recommendation engine (project Module 14).

Takes the measurements produced by measurement_engine and maps them to garment
sizes using anthropometric size charts, with US/UK/EU regional conversion and a
body-shape classification (Module 11). Kept fully rule-based and transparent so
the recommendation is auditable — in production the same interface can be backed
by a learned model trained on real fit-feedback data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Chest circumference (cm) -> alpha size band for tops (shirt / jacket / blazer).
# Bands are inclusive of the lower bound, exclusive of the upper.
_TOP_CHART = [
    ("XS", 0,   86),
    ("S",  86,  94),
    ("M",  94,  102),
    ("L",  102, 110),
    ("XL", 110, 118),
    ("XXL", 118, 128),
    ("XXXL", 128, 999),
]

# Waist circumference (cm) -> numeric trouser waist (inches, EU size).
_TROUSER_CHART = [
    # (label_inch, waist_lo_cm, waist_hi_cm, eu_size)
    ("28", 0,    74,  44),
    ("30", 74,   79,  46),
    ("32", 79,   84,  48),
    ("34", 84,   89,  50),
    ("36", 89,   94,  52),
    ("38", 94,   99,  54),
    ("40", 99,   104, 56),
    ("42", 104,  110, 58),
    ("44", 110,  999, 60),
]

# Regional conversion for alpha top sizes (approximate men's ready-to-wear).
_TOP_REGION = {
    #        US/UK   EU
    "XS":  {"US": "34", "UK": "34", "EU": "44"},
    "S":   {"US": "36", "UK": "36", "EU": "46"},
    "M":   {"US": "38", "UK": "38", "EU": "48"},
    "L":   {"US": "40", "UK": "40", "EU": "50"},
    "XL":  {"US": "42", "UK": "42", "EU": "52"},
    "XXL": {"US": "44", "UK": "44", "EU": "54"},
    "XXXL": {"US": "46", "UK": "46", "EU": "56"},
}


@dataclass
class SizeRecommendation:
    garment: str
    primary_size: str
    regional: dict[str, str]
    confidence: float
    rationale: str
    alternative: Optional[str] = None


def _lookup_top(chest_cm: float) -> tuple[str, str]:
    for label, lo, hi in _TOP_CHART:
        if lo <= chest_cm < hi:
            # flag borderline fits (within 1.5 cm of a boundary) as alternatives
            alt = None
            if chest_cm - lo < 1.5:
                alt = _prev_label(_TOP_CHART, label)
            elif hi - chest_cm < 1.5:
                alt = _next_label(_TOP_CHART, label)
            return label, alt
    return _TOP_CHART[-1][0], None


def _lookup_trouser(waist_cm: float) -> tuple[str, int, str]:
    for label, lo, hi, eu in _TROUSER_CHART:
        if lo <= waist_cm < hi:
            alt = None
            if waist_cm - lo < 1.0:
                alt = _prev_trouser(label)
            elif hi - waist_cm < 1.0:
                alt = _next_trouser(label)
            return label, eu, alt
    last = _TROUSER_CHART[-1]
    return last[0], last[3], None


def _prev_label(chart, label):
    labels = [c[0] for c in chart]
    i = labels.index(label)
    return labels[i - 1] if i > 0 else None


def _next_label(chart, label):
    labels = [c[0] for c in chart]
    i = labels.index(label)
    return labels[i + 1] if i < len(labels) - 1 else None


def _prev_trouser(label):
    labels = [c[0] for c in _TROUSER_CHART]
    i = labels.index(label)
    return labels[i - 1] if i > 0 else None


def _next_trouser(label):
    labels = [c[0] for c in _TROUSER_CHART]
    i = labels.index(label)
    return labels[i + 1] if i < len(labels) - 1 else None


def classify_body_shape(chest_cm: float, waist_cm: float, hip_cm: float) -> str:
    """Simple body-shape classification (Module 11)."""
    if chest_cm == 0 or hip_cm == 0:
        return "unknown"
    if abs(chest_cm - hip_cm) <= 5 and (chest_cm - waist_cm) >= 15:
        return "athletic / inverted-V"
    if hip_cm - chest_cm > 5:
        return "pear / triangle"
    if chest_cm - hip_cm > 5:
        return "inverted-triangle"
    if (chest_cm - waist_cm) < 8:
        return "rectangle / straight"
    return "regular"


def recommend(scan, fit_preference: str = "regular") -> list[SizeRecommendation]:
    """
    Build size recommendations from a ScanResult.
    `fit_preference` in {"slim", "regular", "loose"} nudges the ease allowance.
    """
    ease = {"slim": -2.0, "regular": 0.0, "loose": 3.0}.get(fit_preference, 0.0)

    chest = scan.get_cm("chest")
    waist = scan.get_cm("waist")
    hip = scan.get_cm("hip")

    recs: list[SizeRecommendation] = []

    if chest is not None:
        adj_chest = chest + ease
        size, alt = _lookup_top(adj_chest)
        conf = scan.measurements["chest"].confidence
        for garment in ("shirt", "jacket", "blazer"):
            recs.append(SizeRecommendation(
                garment=garment,
                primary_size=size,
                regional=_TOP_REGION.get(size, {}),
                confidence=conf,
                rationale=f"chest {chest:.1f} cm (+{ease:+.0f} cm {fit_preference} ease) -> {size}",
                alternative=alt,
            ))

    if waist is not None:
        adj_waist = waist + ease
        label, eu, alt = _lookup_trouser(adj_waist)
        conf = scan.measurements["waist"].confidence
        recs.append(SizeRecommendation(
            garment="trouser",
            primary_size=f'W{label}',
            regional={"US": label, "UK": label, "EU": str(eu)},
            confidence=conf,
            rationale=f"waist {waist:.1f} cm (+{ease:+.0f} cm {fit_preference} ease) -> W{label}",
            alternative=(f"W{alt}" if alt else None),
        ))

    return recs
