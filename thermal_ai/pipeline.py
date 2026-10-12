"""Pipeline: picture -> temperature (read if available, predict if not) -> [equipment boxes] -> hotspots -> report.

Three stages (the app calls them separately so you can correct the boxes):
    data = load_data(path, ...)                     temperature grid + how trustworthy it is
    dets, msg = detect_equipment(data)              YOLO boxes (CT, MCB, ...) - optional
    res = analyze_loaded(data, ambient, dets=...)   hotspot + severity + cause per box
`analyze_image(...)` = all three in one call.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detector import Detection, auto_hotspots, get_detector
from .diagnosis import Finding, analyze
from .report import build_pdf
from .temperature import REAL_SOURCES, ThermalData, load_thermal
from .visual import annotate


@dataclass
class Result:
    data: ThermalData
    findings: List[Finding]
    annotated: np.ndarray
    source_text: str
    uncertainty_c: Optional[float]
    scene: Dict[str, float]
    visible: Optional[np.ndarray] = None
    labels: Dict[int, str] = field(default_factory=dict)      # id(finding) -> name shown in tables / report


# ------------------------------------------------------------------ stage 1: load
def load_data(path: str, t_min: float = None, t_max: float = None, unit_hint: str = "F",
              visible_path: Optional[str] = None) -> ThermalData:
    data = load_thermal(path, t_min, t_max, unit_hint)
    if visible_path:                                   # user supplied the original photo
        v = cv2.imread(visible_path)
        if v is not None:
            data.visible = v
    return data


# ------------------------------------------------------------------ stage 2: equipment detection (YOLO)
def detect_equipment(data: ThermalData, weights: str = "models/best.pt",
                     conf: float = 0.25) -> Tuple[List[Detection], str]:
    """Returns (boxes, status message). Never raises: no model / no ultralytics -> ([], reason)."""
    det, why = get_detector(weights, conf)
    if det is None:
        return [], why
    boxes = det.detect(data.display)
    h, w = data.temp.shape
    boxes = [b for b in boxes if (b.bbox[2] - b.bbox[0]) >= 8 and (b.bbox[3] - b.bbox[1]) >= 8]
    msg = f"YOLO found {len(boxes)} item(s)." if boxes else "YOLO found no equipment in this picture."
    return boxes, msg


# ------------------------------------------------------------------ stage 3: analyse
def _region(shape, box, shrink=3):
    h, w = shape
    if box:
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = max(0, x1 + shrink), max(0, y1 + shrink), min(w, x2 - shrink), min(h, y2 - shrink)
        if x2 - x1 > 4 and y2 - y1 > 4:
            return x1, y1, x2, y2
    return 0, 0, w, h


def _pred_src(data: ThermalData) -> str:
    s = "PREDICTED from the colour scale"
    if data.uncertainty_c:
        s += f" (about +-{data.uncertainty_c:.1f} C)"
    return s


def _primary(data: ThermalData, cal: dict) -> dict:
    """The picture's main hotspot: where, how hot, and is the number measured or predicted?"""
    temp = data.temp
    sm = cv2.GaussianBlur(temp, (5, 5), 0)
    x1, y1, x2, y2 = _region(temp.shape, data.box)
    reg = sm[y1:y2, x1:x2]
    iy, ix = np.unravel_index(np.argmax(reg), reg.shape)
    loc = (x1 + int(ix), y1 + int(iy))
    if data.source == "is2_screen" and data.hot_xy:
        loc = data.hot_xy                               # the camera's own Hot point marker
    if data.source in REAL_SOURCES:
        value, src, kind = float(temp[y1:y2, x1:x2].max()), "MEASURED (real temperature data)", "real"
    elif data.onscreen and "max" in data.onscreen:
        value, src, kind = data.onscreen["max"], "MEASURED by the camera (Max number read from the picture)", "camera"
    else:
        value, kind = float(np.percentile(temp[y1:y2, x1:x2], cal.get("q_max", 99.5))), "predicted"
        src = _pred_src(data)
    return dict(region=(x1, y1, x2, y2), loc=loc, value=value, src=src, kind=kind, sm=sm)


def _unknown_mode(data, prim, ambient_c, load_pct, n_extra, cal):
    """No equipment boxes: report the main hotspot + up to n_extra other warm areas as 'unknown'."""
    temp, sm = data.temp, prim["sm"]
    x1, y1, x2, y2 = prim["region"]
    dets = [Detection("unknown", (x1, y1, x2, y2))]
    over, srcs = [prim["value"]], [prim["src"]]
    labels = ["Hotspot 1 (inside camera box)" if data.box else "Hotspot 1 (hottest area)"]
    for d in auto_hotspots(np.nan_to_num(sm, nan=float(np.nanmin(sm))), max_items=n_extra + 3):
        bx1, by1, bx2, by2 = d.bbox
        inter = max(0, min(bx2, x2) - max(bx1, x1)) * max(0, min(by2, y2) - max(by1, y1))
        if inter > 0.3 * (bx2 - bx1) * (by2 - by1):
            continue                                   # already covered by the main hotspot
        dets.append(d); over.append(None)
        srcs.append(_pred_src(data) if not data.radiometric else "MEASURED (real temperature data)")
        labels.append(f"Hotspot {len(dets)} (elsewhere)")
        if len(dets) >= 1 + n_extra:
            break
    findings = analyze(dets, temp, ambient_c, load_pct, tmax_override=over)
    findings[0].hotspot_xy = prim["loc"]
    return findings, srcs, labels


def _equipment_mode(data, boxes, prim, ambient_c, load_pct, cal):
    """Equipment boxes known: one finding per item; same-class items are compared with each other (R/Y/B phases)."""
    temp = data.temp
    h, w = temp.shape
    dets = []
    for b in sorted(boxes, key=lambda d: (d.cls, d.bbox[0])):
        x1, y1, x2, y2 = b.bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 >= 6 and y2 - y1 >= 6:
            dets.append(Detection(b.cls, (x1, y1, x2, y2), b.conf))
    q = cal.get("q_max", 99.5)
    lx, ly = prim["loc"]
    inside = lambda d: d.bbox[0] <= lx <= d.bbox[2] and d.bbox[1] <= ly <= d.bbox[3]
    over, srcs, labels, count = [], [], [], {}
    for d in dets:
        count[d.cls] = count.get(d.cls, 0) + 1
        labels.append(f"{d.cls} {count[d.cls]}")
        if prim["kind"] == "real":
            over.append(None); srcs.append(prim["src"])
        elif prim["kind"] == "camera" and inside(d):
            over.append(prim["value"]); srcs.append(prim["src"])          # the camera's own Max belongs to this box
        else:
            x1, y1, x2, y2 = d.bbox
            over.append(float(np.percentile(temp[y1:y2, x1:x2], q))); srcs.append(_pred_src(data))
    if not any(inside(d) for d in dets):                                   # main hotspot is outside every item
        r = 20
        dets.append(Detection("unknown", (max(0, lx - r), max(0, ly - r), min(w, lx + r), min(h, ly + r))))
        over.append(prim["value"] if prim["kind"] != "real" else None); srcs.append(prim["src"])
        labels.append("Hotspot outside identified equipment")
    findings = analyze(dets, temp, ambient_c, load_pct, tmax_override=over)
    for f, d in zip(findings, dets):
        if inside(d):
            f.hotspot_xy = prim["loc"]
    return findings, srcs, labels


def analyze_loaded(data: ThermalData, ambient_c: float, load_pct: Optional[float] = None,
                   dets: Optional[List[Detection]] = None, n_extra: int = 2) -> Result:
    cal = data.calibration or {}
    prim = _primary(data, cal)
    if dets:
        findings, srcs, labels = _equipment_mode(data, dets, prim, ambient_c, load_pct, cal)
    else:
        findings, srcs, labels = _unknown_mode(data, prim, ambient_c, load_pct, n_extra, cal)
    for f, s in zip(findings, srcs):
        f.temp_source = s
    temp = data.temp
    scene = dict(min=float(np.nanmin(temp)), max=float(np.nanmax(temp)), mean=float(np.nanmean(temp)))
    if data.source == "is2_screen" and data.onscreen:            # camera's own scene min / max are exact
        scene["min"] = data.onscreen.get("min", scene["min"]); scene["max"] = data.onscreen.get("max", scene["max"])
    res = Result(data, findings, annotate(data.display, findings), prim["src"], data.uncertainty_c, scene, data.visible)
    res.labels = {id(f): l for f, l in zip(findings, labels)}
    return res


def analyze_image(path: str, ambient_c: float, load_pct: Optional[float] = None,
                  t_min: float = None, t_max: float = None, unit_hint: str = "F",
                  n_extra: int = 2, visible_path: Optional[str] = None,
                  use_yolo: bool = False, weights: str = "models/best.pt", conf: float = 0.25) -> Result:
    data = load_data(path, t_min, t_max, unit_hint, visible_path)
    dets = detect_equipment(data, weights, conf)[0] if use_yolo else None
    return analyze_loaded(data, ambient_c, load_pct, dets, n_extra)


def make_pdf(res: Result, meta_extra: Dict[str, str]) -> bytes:
    d = res.data
    meta = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        **meta_extra,
        "Temperature source": {"csv": "Real (CSV)", "radiometric": "Real (radiometric file)",
                               "is2": "Real (Fluke .is2 file)",
                               "is2_screen": "Camera numbers (Max/Min/hot point) from .is2 + predicted heat map",
                               "colorbar": ("Camera's own Max/Avg/Min read from picture; heat map predicted from colour bar"
                                            if d.onscreen else "Predicted from colour bar"),
                               "palette": "Predicted from palette", "brightness": "Rough guess"}.get(d.source, d.source),
        "Typical error": f"+-{res.uncertainty_c:.1f} C" if res.uncertainty_c else "real data",
        "Scene (min/mean/max)": f"{res.scene['min']:.1f} / {res.scene['mean']:.1f} / {res.scene['max']:.1f} C",
    }
    return build_pdf(res.findings, d.display, res.annotated, meta, res.labels, {}, visible=res.visible)
