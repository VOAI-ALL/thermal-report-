"""Step 1 of training: make pictures that look exactly like what the app feeds YOLO.
   JPG/PNG with a colour bar -> bar cropped off      .IS2 (Fluke) -> heat-map part of the screen picture
Usage:  python prepare_images.py data/calibration        ->  writes data/to_label/
"""
import sys
from pathlib import Path

import cv2

from thermal_ai.colorbar import find_bar_columns
from thermal_ai.is2 import is2_thermal_crop

src = Path(sys.argv[1] if len(sys.argv) > 1 else "data/calibration")
dst = Path("data/to_label"); dst.mkdir(parents=True, exist_ok=True)
n = 0
for f in sorted(src.glob("*")):
    ext = f.suffix.lower()
    if ext == ".is2":
        img = is2_thermal_crop(str(f))
        if img is None:
            print(f"cannot read {f.name}"); continue
    elif ext in (".jpg", ".jpeg", ".png", ".bmp"):
        img = cv2.imread(str(f))
        try:
            x0, _ = find_bar_columns(img)
            img = img[:, :max(1, x0 - 4)]
        except Exception:
            print(f"no colour bar found in {f.name}, copied unchanged")
    else:
        continue
    cv2.imwrite(str(dst / (f.stem + ".jpg")), img); n += 1
print(f"{n} pictures saved in {dst}  -> upload THIS folder to Roboflow / CVAT for labelling")
