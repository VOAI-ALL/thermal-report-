"""Read newer Fluke .is2 files (e.g. TiS55+) that do NOT contain a usable per-pixel temperature array.

What such a file contains (it is a ZIP archive):
  * Thumbnails/*.jpg        the camera's own screen picture (heat map + Max/Min numbers + colour scale)
  * CalTempDataRex.gpbenc   the camera's exact 256-colour palette (and a raw-count lookup table)
  * Markers/Standard/*      pixel position of the camera's Hot point, Cold point and Centre point
What it does NOT contain: the raw sensor value of every pixel.  So we
  * take the hotspot POSITION from the camera's Hot point marker  (exact),
  * take Max / Min / centre temperatures from the numbers printed on the screen picture (OCR, camera values),
  * PREDICT the rest of the heat map by matching screen colours to the camera's exact palette (+-0.5 C, checked
    on one file only).

Layout constants below are for the 640x480 screen of the Fluke TiS55+. Other models may need new constants.
"""
import re
import zipfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---- screen layout of the TiS55+ at 640x480 (scaled automatically for other sizes)
REF_W, REF_H = 640, 480
THERMAL = (0, 0, 604, 446)                       # x1, y1, x2, y2 of the heat-map area (no bar / status line)
BOX_CENTRE = (255, 12, 385, 60)                  # temperature at the centre cross, top pill
BOX_MAX = (205, 110, 265, 138)
BOX_MIN = (370, 110, 435, 138)
BOX_SCALE_TOP = (565, 8, 630, 46)                # number at the top of the colour bar
BOX_SCALE_BOT = (565, 404, 630, 436)
OVERLAY_BOXES = [(5, 5, 135, 55), (250, 5, 390, 62), (195, 75, 275, 145), (365, 75, 445, 145),
                 (475, 8, 550, 45), (138, 0, 190, 42), (555, 0, 604, 50), (555, 398, 604, 446),
                 (295, 220, 345, 260)]           # logo, pills, HI marker, scale labels, centre cross


# ---------------------------------------------------------------- tiny protobuf reader
def _varint(b: bytes, i: int):
    r = s = 0
    while True:
        x = b[i]; i += 1; r |= (x & 0x7F) << s; s += 7
        if not x & 0x80:
            return r, i


def _parse(b: bytes):
    i, out = 0, []
    while i < len(b):
        k, i = _varint(b, i); f, w = k >> 3, k & 7
        if w == 0:
            v, i = _varint(b, i); out.append((f, w, v))
        elif w == 2:
            n, i = _varint(b, i); out.append((f, w, b[i:i + n])); i += n
        elif w == 5:
            out.append((f, w, b[i:i + 4])); i += 4
        elif w == 1:
            out.append((f, w, b[i:i + 8])); i += 8
        else:
            raise ValueError("unsupported protobuf field")
    return out


def _signed(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


# ---------------------------------------------------------------- helpers
def _palette(z: zipfile.ZipFile) -> Optional[np.ndarray]:
    """256 x 3 BGR palette (cold -> hot) from CalTempDataRex.gpbenc."""
    try:
        top = _parse(z.read("CalTempDataRex.gpbenc"))
        big = _parse([v for f, w, v in top if f == 4 and w == 2][0])
        pal = np.array([_signed(v) & 0xFFFFFFFF for f, w, v in big if f == 12], dtype=np.uint64)
    except Exception:
        return None
    if len(pal) != 256:
        return None
    return np.stack([pal & 0xFF, (pal >> 8) & 0xFF, (pal >> 16) & 0xFF], 1).astype(np.float32)


def _marker(z: zipfile.ZipFile, name: str) -> Optional[Tuple[int, int]]:
    try:
        top = _parse(z.read(f"Markers/Standard/{name}.gpbenc"))
        pos = dict((f, v) for f, w, v in _parse([v for f, w, v in top if f == 1 and w == 2][0]) if w == 0)
        return int(pos[1]), int(pos[2])
    except Exception:
        return None


def _ocr_number(img: np.ndarray, box, thr: int = 190) -> Optional[float]:
    from .ocr import get_pytesseract
    pytesseract = get_pytesseract()
    h, w = img.shape[:2]
    sx, sy = w / REF_W, h / REF_H
    x1, y1, x2, y2 = int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)
    white = (img[y1:y2, x1:x2].min(axis=2) >= thr).astype(np.uint8) * 255
    m = cv2.resize(255 - white, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    txt = pytesseract.image_to_string(m, config="--psm 7 -c tessedit_char_whitelist=0123456789.-").strip()
    mt = re.search(r"-?\d{1,3}\.\d", txt)
    if mt:
        return float(mt.group())
    mt = re.search(r"(-?\d{1,3})(\d)\b", txt)                 # decimal point lost by OCR
    return float(f"{mt.group(1)}.{mt.group(2)}") if mt else None


def read_is2_visible(path: str) -> Optional[np.ndarray]:
    """The ordinary (visible-light) photo stored inside the .is2 file, if any."""
    try:
        z = zipfile.ZipFile(path)
        names = [n for n in z.namelist()
                 if n.lower().startswith("images/") and n.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not names:
            return None
        best = max(names, key=lambda n: z.getinfo(n).file_size)
        return cv2.imdecode(np.frombuffer(z.read(best), np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def is2_thermal_crop(path: str) -> Optional[np.ndarray]:
    """The heat-map part of the screen picture (exactly what the app shows YOLO). No OCR needed."""
    try:
        z = zipfile.ZipFile(path)
        t = [n for n in z.namelist() if n.lower().startswith("thumbnails/") and n.lower().endswith(".jpg")][0]
        img = cv2.imdecode(np.frombuffer(z.read(t), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        sx, sy = w / REF_W, h / REF_H
        return img[int(THERMAL[1] * sy):int(THERMAL[3] * sy), int(THERMAL[0] * sx):int(THERMAL[2] * sx)].copy()
    except Exception:
        return None


# ---------------------------------------------------------------- main entry
def read_is2_screen(path: str) -> Dict:
    """Returns dict: display (BGR), temp (deg C, predicted), onscreen {max,min,centre}, hot_xy, cold_xy, centre_xy, scale."""
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise ValueError("This .is2 file is not a ZIP archive - unknown Fluke format.")
    thumbs = [n for n in z.namelist() if n.lower().startswith("thumbnails/") and n.lower().endswith(".jpg")]
    if not thumbs:
        raise ValueError("No screen picture inside this .is2 file.")
    img = cv2.imdecode(np.frombuffer(z.read(thumbs[0]), np.uint8), cv2.IMREAD_COLOR)
    pal = _palette(z)
    if pal is None:
        raise ValueError("No colour palette found inside this .is2 file.")

    # numbers printed by the camera
    t_top, t_bot = _ocr_number(img, BOX_SCALE_TOP), _ocr_number(img, BOX_SCALE_BOT)
    mx, mn, ce = _ocr_number(img, BOX_MAX), _ocr_number(img, BOX_MIN), _ocr_number(img, BOX_CENTRE)
    if t_top is None or t_bot is None:                        # auto-range: scale ends = Max / Min
        t_top, t_bot = mx, mn
    if t_top is None or t_bot is None or t_top <= t_bot:
        raise ValueError("Could not read the temperature scale from this .is2 screen picture.")
    onscreen = {k: v for k, v in (("max", mx), ("min", mn), ("centre", ce)) if v is not None}

    # marker positions are on the IR grid; its size = 2 x centre marker
    centre = _marker(z, "Centerpoint")
    hot, cold = _marker(z, "Hotpoint"), _marker(z, "Coldpoint")
    h, w = img.shape[:2]
    ir_w, ir_h = (centre[0] * 2, centre[1] * 2) if centre else (w, h)
    sc = lambda p: None if p is None else (int(round(p[0] * w / ir_w)), int(round(p[1] * h / ir_h)))

    # predicted heat map: screen colours -> exact camera palette -> temperature
    from .colorbar import _fill_masked, _match_colors
    sx, sy = w / REF_W, h / REF_H
    x1, y1, x2, y2 = int(THERMAL[0] * sx), int(THERMAL[1] * sy), int(THERMAL[2] * sx), int(THERMAL[3] * sy)
    region = img[y1:y2, x1:x2]
    lut_t = np.linspace(t_bot, t_top, 256).astype(np.float32)
    temp, _ = _match_colors(region, pal, lut_t)
    mask = np.zeros(region.shape[:2], np.uint8)
    for bx in OVERLAY_BOXES:
        mask[max(0, int(bx[1] * sy) - y1):max(0, int(bx[3] * sy) - y1),
             max(0, int(bx[0] * sx) - x1):max(0, int(bx[2] * sx) - x1)] = 1
    mn_ch, mx_ch = region.min(axis=2), region.max(axis=2)
    white = ((mn_ch > 200) & ((mx_ch.astype(int) - mn_ch) < 60)).astype(np.uint8)
    solid = cv2.dilate(cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), np.ones((5, 5), np.uint8))
    mask |= ((white > 0) & (solid == 0)).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    temp = _fill_masked(temp, mask > 0)
    temp = np.clip(cv2.GaussianBlur(temp, (0, 0), 2), t_bot, t_top).astype(np.float32)

    def clamp(p):
        return None if p is None else (min(max(p[0], 0), x2 - x1 - 1), min(max(p[1], 0), y2 - y1 - 1))
    return dict(display=region.copy(), temp=temp, onscreen=onscreen, scale=(t_bot, t_top),
                hot_xy=clamp(sc(hot)), cold_xy=clamp(sc(cold)), centre_xy=clamp(sc(centre)),
                visible=read_is2_visible(path))
