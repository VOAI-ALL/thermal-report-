"""Equipment detection (YOLO) + a no-model fallback that finds hot blobs."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from .config import CLASSES


@dataclass
class Detection:
    cls: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    conf: float = 1.0


class EquipmentDetector:
    def __init__(self, weights: str, conf: float = 0.25):
        self.available = False
        self.model = None
        self.conf = conf
        if weights and Path(weights).exists():
            from ultralytics import YOLO
            self.model = YOLO(weights)
            self.available = True

    def detect(self, img_bgr: np.ndarray) -> List[Detection]:
        if not self.available:
            return []
        res = self.model.predict(img_bgr, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            name = res.names[int(b.cls[0])]
            out.append(Detection(name, (x1, y1, x2, y2), float(b.conf[0])))
        return out


def to_yolo_lines(dets: List[Detection], w: int, h: int) -> List[str]:
    """YOLO label lines: class_id x_center y_center width height (all 0..1)."""
    out = []
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        out.append(f"{CLASSES.index(d.cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                   f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return out


_CACHE = {}


def get_detector(weights: str, conf: float = 0.25):
    """Returns (detector, '') or (None, reason). Loads the model once and reuses it."""
    if not weights or not Path(weights).exists():
        return None, f"No trained model at {weights} - train one (see README, Phase 2) or add boxes by hand."
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return None, "The 'ultralytics' package is not installed:  pip install ultralytics"
    key = str(Path(weights).resolve())
    if key not in _CACHE:
        try:
            _CACHE[key] = EquipmentDetector(weights, conf)
        except Exception as e:
            return None, f"Could not load the model ({str(e)[:100]})."
    det = _CACHE[key]
    det.conf = conf
    return det, ""


def auto_hotspots(temp: np.ndarray, max_items: int = 6, pad: int = 12) -> List[Detection]:
    """Fallback when no trained model exists: box the hottest blobs as 'unknown' equipment."""
    thr = max(np.percentile(temp, 98), temp.mean() + 2 * temp.std())
    mask = (temp >= thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    blobs = sorted(range(1, n), key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:max_items]
    h, w = temp.shape
    out = []
    for i in blobs:
        x, y, bw, bh = [int(v) for v in stats[i, :4]]
        out.append(Detection(
            "unknown",
            (max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)),
        ))
    return out
