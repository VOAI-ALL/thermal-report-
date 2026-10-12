"""Estimate temperatures from a COLOUR-ONLY thermal image by reading its colour scale bar.

How it works (no guessing from brightness):
  1. Find the vertical colour bar (smooth, tall gradient) on the right side of the image.
  2. OCR the numbers printed beside it (e.g. 94.1 ... 89.9) and their pixel rows.
  3. Fit a straight line  temperature = a * row + b  through those labels.
  4. Build a lookup table: bar colour at each row -> temperature.
  5. For every pixel of the thermal picture, find the closest bar colour -> temperature.
  6. Remove the on-screen overlay (text, box lines, crosshairs) so it is not read as heat.

Needs the `tesseract` program + `pytesseract` package for step 2.
If OCR is not available, pass `scale=(t_bottom, t_top)` (the min/max printed on the bar).
"""
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ColorBar:
    x0: int                 # first / last bar column used (inner columns, borders removed)
    x1: int
    y_top: float            # pixel row of the top / bottom scale labels
    y_bot: float
    t_top: float            # temperature at those rows (in `unit`)
    t_bot: float
    slope: float            # temperature change per pixel row
    intercept: float
    unit: str               # "F" or "C"
    n_labels: int           # how many labels the OCR read (more = more reliable)
    residual: float         # max error of the line fit over the labels (unit)


# ------------------------------------------------------------------ bar detection
def find_bar_columns(img: np.ndarray) -> Tuple[int, int]:
    """Columns that form a tall smooth gradient (large range down the column, small row-to-row change)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    ok = np.zeros(w, bool)
    for x in range(int(w * 0.5), w):
        col = g[:, x]
        ok[x] = col.std() > 55 and np.abs(np.diff(col)).mean() < 9
    runs, start = [], None
    for x in range(w + 1):
        v = ok[x] if x < w else False
        if v and start is None:
            start = x
        elif not v and start is not None:
            runs.append((start, x - 1)); start = None
    runs = [r for r in runs if r[1] - r[0] + 1 >= 6]
    if not runs:
        raise ValueError("Could not find a colour scale bar in this image.")
    x0, x1 = max(runs, key=lambda r: r[1] - r[0])
    return x0 + 2, x1 - 2          # trim borders


# ------------------------------------------------------------------ OCR of the scale
def _ocr_labels(img: np.ndarray, bar_x1: int) -> Tuple[List[Tuple[float, float]], Optional[str]]:
    from .ocr import get_pytesseract
    pytesseract = get_pytesseract()
    h, w = img.shape[:2]
    x_start = bar_x1 + 5
    reg = img[:, x_start:min(w, x_start + 100)]
    gray = cv2.cvtColor(reg, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    d = pytesseract.image_to_data(
        th, config="--psm 6 -c tessedit_char_whitelist=0123456789.-",
        output_type=pytesseract.Output.DICT)
    labels = []
    for t, y, hh in zip(d["text"], d["top"], d["height"]):
        t = t.strip()
        if re.fullmatch(r"-?\d{1,3}\.\d", t):
            labels.append((float(t), (y + hh / 2) / 4.0))

    unit = None
    if labels:
        y_last = max(y for _, y in labels)
        below = gray[int(y_last) + 6:, :]
        if below.size:
            b = cv2.resize(below, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            _, bt = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for psm in (7, 6):
                txt = pytesseract.image_to_string(bt, config=f"--psm {psm}")
                if re.search(r"F", txt):
                    unit = "F"; break
                if re.search(r"C", txt):
                    unit = "C"; break
    return labels, unit


def _fit_line(labels: List[Tuple[float, float]]):
    """value = slope*row + intercept, rejecting OCR misreads."""
    v = np.array([l[0] for l in labels]); y = np.array([l[1] for l in labels])
    keep = np.ones(len(v), bool)
    for _ in range(4):
        if keep.sum() < 3:
            break
        a, b = np.polyfit(y[keep], v[keep], 1)
        res = np.abs(v - (a * y + b))
        tol = max(0.08 * abs(a * (y.max() - y.min())), 0.05)   # 8 % of the span
        new = res <= tol
        if (new == keep).all():
            break
        keep = new
    a, b = np.polyfit(y[keep], v[keep], 1)
    resid = float(np.abs(v[keep] - (a * y[keep] + b)).max())
    return a, b, resid, int(keep.sum()), y[keep]


def read_colorbar(img: np.ndarray, scale: Optional[Tuple[float, float]] = None,
                  unit_hint: str = "F") -> ColorBar:
    x0, x1 = find_bar_columns(img)
    labels, unit = ([], None)
    from .ocr import TesseractMissing
    try:
        labels, unit = _ocr_labels(img, x1 + 2)
    except TesseractMissing:
        raise
    except Exception:
        pass
    if len(labels) >= 3:
        a, b, resid, n, ys = _fit_line(labels)
        y_top, y_bot = float(ys.min()), float(ys.max())
        return ColorBar(x0, x1, y_top, y_bot, a * y_top + b, a * y_bot + b, a, b,
                        unit or unit_hint, n, resid)
    if scale is None:
        raise ValueError("Could not read the scale numbers (OCR failed). "
                         "Give scale=(min, max) as printed beside the colour bar.")
    # manual: assume bar spans from first to last non-background row
    g = cv2.cvtColor(img[:, x0:x1 + 1], cv2.COLOR_BGR2GRAY).mean(axis=1)
    rows = np.where(g < 250)[0]
    y_top, y_bot = float(rows.min()) + 9, float(rows.max()) - 8
    t_min, t_max = scale
    a = (t_min - t_max) / (y_bot - y_top)
    b = t_max - a * y_top
    return ColorBar(x0, x1, y_top, y_bot, t_max, t_min, a, b, unit_hint, 0, 0.0)


# ------------------------------------------------------------------ pixel -> temperature
def _lab(bgr: np.ndarray) -> np.ndarray:
    f = bgr.astype(np.float32) / 255.0
    return cv2.cvtColor(f.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)


def _fill_masked(temp: np.ndarray, mask: np.ndarray, k: int = 11) -> np.ndarray:
    """Replace masked pixels by the average of nearby VALID pixels (normalised convolution)."""
    out = temp.copy()
    valid = (~mask).astype(np.float32)
    for _ in range(80):
        if valid.min() > 0:
            break
        num = cv2.blur(out * valid, (k, k)); den = cv2.blur(valid, (k, k))
        fill = np.where(den > 0.05, num / np.maximum(den, 1e-6), out)
        out = np.where(valid > 0, out, fill).astype(np.float32)
        valid = np.where((valid > 0) | (den > 0.05), 1.0, 0.0).astype(np.float32)
    return out


def _match_colors(region: np.ndarray, lut_bgr: np.ndarray, lut_t: np.ndarray):
    """Nearest bar colour (Lab distance) for every pixel -> temperature, plus the colour distance."""
    lut_lab = _lab(lut_bgr)
    pix_lab = _lab(region.reshape(-1, 3))
    idx = np.empty(len(pix_lab), np.int32); dist = np.empty(len(pix_lab), np.float32)
    for s in range(0, len(pix_lab), 20000):
        dd = ((pix_lab[s:s + 20000, None, :] - lut_lab[None, :, :]) ** 2).sum(-1)
        idx[s:s + 20000] = dd.argmin(1); dist[s:s + 20000] = np.sqrt(dd.min(1))
    return lut_t[idx].reshape(region.shape[:2]).astype(np.float32), dist.reshape(region.shape[:2])


def _remove_overlay(region: np.ndarray, temp: np.ndarray, dist: np.ndarray, lut_t: np.ndarray) -> np.ndarray:
    """Remove camera text / box lines / crosshairs / timestamp and fill the holes smoothly."""
    mn = region.min(axis=2); mx = region.max(axis=2)
    whiteish = ((mn > 200) & ((mx.astype(int) - mn) < 60)).astype(np.uint8)
    solid = cv2.morphologyEx(whiteish, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    solid = cv2.dilate(solid, np.ones((5, 5), np.uint8))       # real hot blobs (>=5 px) are kept
    thin_white = (whiteish > 0) & (solid == 0)                  # thin strokes = overlay
    overlay = (thin_white | (dist > 22)).astype(np.uint8)       # dist>22: colours not on the bar
    overlay = cv2.dilate(overlay, np.ones((3, 3), np.uint8), iterations=2)
    clean = _fill_masked(temp, overlay > 0) if overlay.any() else temp
    return np.clip(clean, lut_t.min(), lut_t.max()).astype(np.float32)


def estimate_temperature(img: np.ndarray, bar: ColorBar):
    """Returns (temp_in_bar_unit [HxW float32], overlay_mask [HxW bool], thermal_x_end)."""
    h, w = img.shape[:2]
    ys = np.arange(int(round(bar.y_top)), int(round(bar.y_bot)) + 1)
    lut_bgr = np.array([np.median(img[y, bar.x0:bar.x1 + 1], axis=0) for y in ys])
    lut_t = bar.slope * ys + bar.intercept
    x_end = max(1, bar.x0 - 4)               # thermal picture = everything left of the bar
    region = img[:, :x_end]
    temp, dist = _match_colors(region, lut_bgr, lut_t)
    clean = _remove_overlay(region, temp, dist, lut_t)
    full = np.full((h, w), np.nan, np.float32); full[:, :x_end] = clean
    ov = np.ones((h, w), bool); ov[:, :x_end] = False
    return full, ov, x_end


# ------------------------------------------------------------------ palette (learned from many images)
def extract_palette(img: np.ndarray, bar: ColorBar, n: int = 256) -> np.ndarray:
    """Bar colours resampled on a 0 (cold) .. 1 (hot) grid, shape (n, 3) BGR."""
    ys = np.arange(int(round(bar.y_top)), int(round(bar.y_bot)) + 1)
    cols = np.array([np.median(img[y, bar.x0:bar.x1 + 1], axis=0) for y in ys], np.float32)
    u = (ys - bar.y_bot) / (bar.y_top - bar.y_bot)           # 0 at bottom label, 1 at top label
    order = np.argsort(u); u, cols = u[order], cols[order]
    grid = np.linspace(0, 1, n)
    return np.stack([np.interp(grid, u, cols[:, c]) for c in range(3)], axis=1)


def load_palette(path: str = "models/palette.json"):
    import json, os
    if not os.path.exists(path):
        return None
    return np.array(json.load(open(path))["colors"], np.float32)


def estimate_with_palette(img: np.ndarray, t_min: float, t_max: float, palette: np.ndarray):
    """For pictures WITHOUT a colour bar: use the palette learned from your other pictures
    and the scale (t_min, t_max) given by the user. Returns temperature map in the user's unit."""
    lut_t = np.linspace(t_min, t_max, len(palette)).astype(np.float32)
    temp, dist = _match_colors(img, palette, lut_t)
    return _remove_overlay(img, temp, dist, lut_t)


def load_calibration(path: str = "models/colorbar_calibration.json"):
    import json, os
    return json.load(open(path)) if os.path.exists(path) else None


def f_to_c(t):
    return (t - 32.0) * 5.0 / 9.0


def apply_calibration(temp: np.ndarray, bar: ColorBar, path: str = "models/colorbar_calibration.json"):
    """Apply the correction learned by calibrate.py (if the file exists). Works in scale-position units."""
    import json, os
    if not os.path.exists(path):
        return temp, False
    cal = json.load(open(path))
    span = bar.t_top - bar.t_bot
    u = (temp - bar.t_bot) / span
    return (bar.t_bot + np.polyval(cal["coef"], u) * span).astype(np.float32), True
