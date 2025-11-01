# Palette 1: Vibrant & High-Contrast (Unchanged)
from typing import Dict, Tuple


CATEGORY_COLORS: Dict[str, Tuple[float, float, float]] = {
    # Red – warnings / problems
    "limitation":   (0.95, 0.25, 0.25),   # #F24040

    # Green – positive outcomes
    "results":      (0.20, 0.75, 0.30),   # #33BF4D

    # Blue – technical / method
    "method":       (0.15, 0.45, 0.85),   # #2673D9

    # Orange – future / open
    "future_work":  (1.00, 0.60, 0.20),   # #FF9933

    # Purple – novelty / creativity
    "innovation":   (0.65, 0.30, 0.95),   # #A64DFF

    # Teal – data
    "dataset":      (0.00, 0.65, 0.65),   # #00A6A6

    # Cyan – reasoning / explanation
    "reason":       (0.30, 0.80, 0.90),   # #4DCCCC

    # Gray‑blue – background literature
    "related_work": (0.45, 0.60, 0.80),   # #7399CC
}

LABELS = [
    "innovation", "related work", "limitation", "method",
    "results", "dataset", "future_work",
    "reason", "none"
]