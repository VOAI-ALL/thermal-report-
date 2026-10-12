"""Step 3: put your labelled export into the folders YOLO needs (80 % train / 20 % val).
Usage:  python split_dataset.py path/to/roboflow_export_folder
Works with a flat folder (image + same-name .txt) or Roboflow's train/valid/test sub-folders.
Pictures of the same panel (same name before '_IR_') always stay in the same split.
"""
import random, shutil, sys
from collections import defaultdict
from pathlib import Path

IMG = {".jpg", ".jpeg", ".png", ".bmp"}
src = Path(sys.argv[1]); out = Path("data/dataset")
images = [p for p in src.rglob("*") if p.suffix.lower() in IMG]
labels = {p.stem: p for p in src.rglob("*.txt") if p.name.lower() not in ("readme.txt", "classes.txt")}
pairs = [(i, labels[i.stem]) for i in images if i.stem in labels]
print(f"{len(images)} images, {len(pairs)} with labels")
if len(pairs) < 5:
    sys.exit("Too few labelled images. Label them first (Roboflow / CVAT) and export in YOLOv8 format.")

groups = defaultdict(list)
for im, lb in pairs:
    groups[im.stem.split("_IR_")[0]].append((im, lb))
keys = sorted(groups); random.Random(0).shuffle(keys)
n_val = max(1, round(len(keys) * 0.2))
split = {k: ("val" if i < n_val else "train") for i, k in enumerate(keys)}

for s in ("train", "val"):
    for sub in ("images", "labels"):
        d = out / sub / s
        if d.exists(): shutil.rmtree(d)
        d.mkdir(parents=True)
bad, count = 0, {"train": 0, "val": 0}
for k, items in groups.items():
    s = split[k]
    for im, lb in items:
        for line in lb.read_text().split("\n"):
            if line.strip() and not (0 <= int(line.split()[0]) <= 7):
                bad += 1
        shutil.copy(im, out / "images" / s / im.name); shutil.copy(lb, out / "labels" / s / lb.name)
        count[s] += 1
print(f"train {count['train']}  val {count['val']}  -> {out}")
if bad: print(f"WARNING: {bad} label lines use a class number outside 0-7. Check class order against data.yaml!")
