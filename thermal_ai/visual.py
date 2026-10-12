"""Draw boxes, hotspots and labels on the image."""
from typing import List

import cv2
import numpy as np

from .diagnosis import Finding

COLORS = {0: (80, 200, 80), 1: (0, 220, 255), 2: (0, 140, 255), 3: (0, 0, 255)}  # BGR


def annotate(display: np.ndarray, findings: List[Finding]) -> np.ndarray:
    """Boxes are numbered by priority (worst first) - the same # used in the PDF table."""
    img = display.copy()
    scale = max(img.shape[1] / 640, 0.6)
    ranked = sorted(findings, key=lambda f: (-f.severity, -f.max_temp))
    for n, f in enumerate(ranked, 1):
        col = COLORS[f.severity]
        x1, y1, x2, y2 = f.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), col, max(1, int(2 * scale)))
        hx, hy = f.hotspot_xy
        r = int(6 * scale)
        cv2.line(img, (hx - r, hy), (hx + r, hy), (255, 255, 255), 1)
        cv2.line(img, (hx, hy - r), (hx, hy + r), (255, 255, 255), 1)
        label = f"{n}: {f.max_temp:.1f}C" if f.equipment == "unknown" else f"{n}: {f.equipment} {f.max_temp:.1f}C"
        fs, th = 0.45 * scale, max(1, int(scale))
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        ty = y1 - 4 if y1 - tht - 6 > 0 else y2 + tht + 4
        cv2.rectangle(img, (x1, ty - tht - 3), (x1 + tw + 4, ty + 3), (0, 0, 0), -1)
        cv2.putText(img, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
    return img
