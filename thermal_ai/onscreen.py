"""Read what the camera printed on the picture: measurement box + Max / Avg / Min numbers.
These are used as GROUND TRUTH to check and calibrate the colour-bar temperatures."""
import re
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def find_box(img: np.ndarray, x_end: int) -> Optional[Tuple[int, int, int, int]]:
    """White rectangle drawn by the camera. Returns x1, y1, x2, y2 or None."""
    mn = img[:, :x_end].min(axis=2)
    white = (mn >= 200).astype(np.uint8)
    hor = cv2.morphologyEx(white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)))
    ver = cv2.morphologyEx(white, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30)))
    ys, xs = np.where((hor | ver) > 0)
    if len(xs) < 50:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def read_stats(img: np.ndarray, x_end: int) -> Dict[str, float]:
    """OCR the 'Max / Avg / Min' text (white text near the top of the picture)."""
    from .ocr import get_pytesseract
    pytesseract = get_pytesseract()
    mn = img[:, :x_end].min(axis=2)
    m = (mn >= 170).astype(np.uint8)[0:int(img.shape[0] * 0.25), :]
    inv = cv2.resize(255 - m * 255, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    inv = cv2.GaussianBlur(inv, (3, 3), 0)
    txt = pytesseract.image_to_string(
        inv, config="--psm 6 -c tessedit_char_whitelist=MaxAvgMin:0123456789.")
    out = {}
    for name in ("Max", "Avg", "Min"):
        mt = re.search(name + r"\s*:?\s*(\d{2,3})\.?(\d)\b", txt)
        if mt:                                     # decimal point is often lost -> put it back
            out[name.lower()] = float(f"{mt.group(1)}.{mt.group(2)}")
    return out
