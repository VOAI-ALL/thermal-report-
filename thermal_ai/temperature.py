"""Get a temperature map (deg C) from a thermal file - best available source first.

  1. CSV exported from camera software                -> real temperatures
  1b. Fluke .is2 file (SmartView)                     -> real temperatures (needs: pip install fluke-thermal-reader)
  2. Radiometric FLIR JPG                             -> real temperatures
  3. Colour picture WITH a colour scale bar           -> predicted from the bar (+ learned correction)
     (+ the camera's own Max/Avg/Min numbers are read from the picture when present)
  4. Colour picture WITHOUT a bar                     -> predicted with the learned palette + your min/max
  5. Anything else                                    -> rough guess from brightness
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


REAL_SOURCES = ("csv", "radiometric", "is2")   # temperatures that are measurements, not predictions


@dataclass
class ThermalData:
    temp: np.ndarray                  # HxW float32, deg C
    display: np.ndarray               # HxWx3 BGR picture used for drawing
    radiometric: bool                 # True = temperatures are real measurements
    note: str
    source: str = ""                  # csv | radiometric | colorbar | palette | brightness
    onscreen: Optional[Dict[str, float]] = None      # camera's own Max/Avg/Min in deg C (if printed on picture)
    box: Optional[Tuple[int, int, int, int]] = None  # camera's measurement box
    uncertainty_c: Optional[float] = None            # typical +- error in deg C
    calibration: Optional[dict] = None
    hot_xy: Optional[Tuple[int, int]] = None      # camera's own Hot point marker (pixel in `display`)
    cold_xy: Optional[Tuple[int, int]] = None
    centre_xy: Optional[Tuple[int, int]] = None
    visible: Optional[np.ndarray] = None          # original visible-light photo, if the file has one


def colorize(temp: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(temp)), float(np.nanmax(temp))
    norm = ((temp - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


def _read_csv(path: str) -> np.ndarray:
    with open(path, "r", errors="ignore") as f:
        head = f.read(4096)
    sep = ";" if head.count(";") > head.count(",") else ("\t" if head.count("\t") > head.count(",") else ",")
    df = pd.read_csv(path, header=None, sep=sep, engine="python")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all").dropna(axis=1, how="all")
    return df.to_numpy(dtype=np.float32)


def _load_is2(path: str) -> ThermalData:
    """1) exact per-pixel temperatures via fluke-thermal-reader (older / supported cameras)
       2) otherwise the built-in reader: camera's own Max/Min/hot point + predicted heat map (newer cameras)"""
    pkg_error = None
    try:
        from fluke_thermal_reader import read_is2
        d = read_is2(path)
        temp = np.asarray(d["data"], dtype=np.float32)
        if temp.ndim == 2 and temp.size >= 100 and np.isfinite(temp).any():
            temp = np.where(np.isfinite(temp), temp, np.nanmedian(temp)).astype(np.float32)
            if -60 <= temp.min() and temp.max() <= 2000:
                model = d.get("CameraModel") if isinstance(d, dict) else None
                from .is2 import read_is2_visible
                return ThermalData(temp, colorize(temp), True,
                                   "Real temperatures read from the Fluke .is2 file" + (f" ({model})." if model else "."),
                                   source="is2", visible=read_is2_visible(path))
        pkg_error = "unexpected temperature data"
    except ImportError:
        pkg_error = "package fluke-thermal-reader not installed"
    except Exception as e:
        pkg_error = str(e)[:120]

    from .ocr import TesseractMissing
    try:
        from .is2 import read_is2_screen
        r = read_is2_screen(path)
    except TesseractMissing:
        raise
    except Exception as e:
        raise ValueError(f"Could not read this .is2 file ({e}). Export it as CSV from Fluke SmartView and upload the CSV.")
    note = ("This .is2 file has no per-pixel temperature array (newer Fluke format). The hot point position and the Max/Min "
            "numbers are the CAMERA'S OWN values; the rest of the heat map is PREDICTED from the screen picture "
            f"(rough, about +-0.5 C). [{pkg_error}]")
    return ThermalData(r["temp"], r["display"], False, note, source="is2_screen", onscreen=r["onscreen"],
                       uncertainty_c=0.6, hot_xy=r["hot_xy"], cold_xy=r["cold_xy"], centre_xy=r["centre_xy"],
                       visible=r.get("visible"))


def load_thermal(path: str, t_min: float = None, t_max: float = None, unit_hint: str = "F") -> ThermalData:
    """t_min / t_max = scale printed beside the colour bar, only needed if there is no readable bar."""
    p = Path(path)
    if p.suffix.lower() == ".is2":
        return _load_is2(path)
    if p.suffix.lower() == ".is3":
        raise ValueError(".is3 (Fluke video) files are not supported. Open it in SmartView and export a frame as .is2 or CSV.")
    if p.suffix.lower() in (".csv", ".txt"):
        temp = _read_csv(path)
        return ThermalData(temp, colorize(temp), True, "Real temperatures loaded from CSV.", source="csv")

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    # 2) radiometric FLIR JPG
    try:
        import flirimageextractor
        flir = flirimageextractor.FlirImageExtractor()
        flir.process_image(path)
        temp = flir.get_thermal_np().astype(np.float32)
        vis = None
        try:
            rgb = flir.get_rgb_np()
            vis = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        except Exception:
            pass
        return ThermalData(temp, cv2.resize(img, (temp.shape[1], temp.shape[0])), True,
                           "Real temperatures extracted from FLIR radiometric data.", source="radiometric", visible=vis)
    except Exception:
        pass

    from . import colorbar as cb
    from .ocr import TesseractMissing
    cal = cb.load_calibration()

    # 3) colour picture with a colour scale bar
    try:
        bar = cb.read_colorbar(img, scale=(t_min, t_max) if t_min is not None else None, unit_hint=unit_hint)
        t, _, xe = cb.estimate_temperature(img, bar)
        t, calibrated = cb.apply_calibration(t, bar)
        t = t[:, :xe]
        conv = (lambda v: cb.f_to_c(v)) if bar.unit == "F" else (lambda v: v)
        onscreen, box = None, None
        try:
            from .onscreen import find_box, read_stats
            box = find_box(img, xe)
            st = read_stats(img, xe)
            onscreen = {k: float(conv(v)) for k, v in st.items()} or None
        except Exception:
            pass
        unc = (cal or {}).get("unc_c", {}).get("max") if calibrated else None
        if unc is None:
            unc = 0.6 if bar.unit == "F" else 0.35          # ~1 F / ~0.6 F default before learning
        note = (f"Temperatures PREDICTED from the colour scale bar ({bar.t_bot:.1f}-{bar.t_top:.1f} deg {bar.unit}, "
                f"{bar.n_labels} labels read, "
                f"{'with learned correction' if calibrated else 'NO learned correction yet - run learn_temperature.py'}).")
        return ThermalData(conv(t).astype(np.float32), img[:, :xe].copy(), False, note, source="colorbar",
                           onscreen=onscreen, box=box, uncertainty_c=float(unc), calibration=cal)
    except TesseractMissing:
        raise
    except Exception:
        pass

    # 4) colour picture WITHOUT bar: learned palette + user's scale
    pal = cb.load_palette()
    if pal is not None and t_min is not None and t_max is not None:
        t = cb.estimate_with_palette(img, t_min, t_max, pal)
        if unit_hint == "F":
            t = cb.f_to_c(t)
        return ThermalData(t.astype(np.float32), img.copy(), False,
                           "Temperatures PREDICTED with the learned colour palette and the scale you typed. Less reliable.",
                           source="palette", uncertainty_c=1.0, calibration=cal)

    # 5) last resort
    if t_min is None or t_max is None:
        raise ValueError("No colour scale bar found. Type the scale min/max, export a CSV, or use a radiometric file.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    temp = t_min + gray * (t_max - t_min)
    if unit_hint == "F":
        temp = (temp - 32) * 5 / 9
    return ThermalData(temp, img, False, "ROUGH guess from brightness only. Not reliable.",
                       source="brightness", uncertainty_c=3.0)
