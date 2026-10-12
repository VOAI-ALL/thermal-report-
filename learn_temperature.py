"""THE LEARNING STEP.  Teaches the AI the colour -> temperature pattern from YOUR pictures.

    python learn_temperature.py data/calibration            (add --degree 2 for a curved correction)

Use pictures that show the colour bar AND the camera's own numbers (Max / Avg / Min in a box).
For each picture it reads those numbers as the ground truth and compares them with the temperature
it estimated from the colours. From all pictures it learns:
   1. a correction curve  (estimated -> true), so colour readings are not biased
   2. which percentile of the hot area best matches the camera's Max and Min (noise-robust hotspot)
   3. the typical error (+-), printed in every report
   4. the camera's colour palette (so pictures WITHOUT a bar can still be read when you give min/max)
Saved in models/. Run it again whenever you add more pictures - it gets better with more data.
If OCR misreads a value, add <folder>/labels.csv with columns: filename,max,avg,min
"""
import argparse, csv, json, os
from pathlib import Path

import cv2
import numpy as np

from thermal_ai.colorbar import estimate_temperature, extract_palette, read_colorbar
from thermal_ai.onscreen import find_box, read_stats

QMAX = [90, 95, 97, 98, 99, 99.5, 99.9, 100]
QMIN = [0, 0.1, 0.5, 1, 2, 3, 5, 10]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--degree", type=int, default=1)
    a = ap.parse_args(); folder = Path(a.folder)
    manual = {}
    if (folder / "labels.csv").exists():
        for r in csv.DictReader(open(folder / "labels.csv")):
            manual[r["filename"]] = {k: float(r[k]) for k in ("max", "avg", "min") if r.get(k)}

    recs, palettes = [], []
    for f in sorted(folder.glob("*")):
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        img = cv2.imread(str(f))
        try:
            bar = read_colorbar(img)
            palettes.append(extract_palette(img, bar))
            t, _, xe = estimate_temperature(img, bar)
            box = find_box(img, xe)
            st = manual.get(f.name) or read_stats(img, xe)
            if box is None or "avg" not in st:
                raise ValueError("camera box or Avg number not found")
        except Exception as e:
            print(f"SKIP {f.name}: {e}"); continue
        x1, y1, x2, y2 = box
        recs.append(dict(name=f.name, unit=bar.unit, tb=bar.t_bot, span=bar.t_top - bar.t_bot,
                         inside=t[y1 + 3:y2 - 2, x1 + 3:x2 - 2].ravel(), true=st))
    print(f"\nUsable pictures: {len(recs)}")
    if palettes:
        os.makedirs("models", exist_ok=True)
        json.dump({"colors": np.median(np.array(palettes), axis=0).round(1).tolist(), "n_images": len(palettes)},
                  open("models/palette.json", "w"))
        print(f"Palette learned from {len(palettes)} pictures -> models/palette.json")
    if len(recs) < 5:
        print("Need at least 5 usable pictures to learn the correction."); return

    to_c = lambda r: 5 / 9 if r["unit"] == "F" else 1.0
    u_est = np.array([(r["inside"].mean() - r["tb"]) / r["span"] for r in recs])
    u_true = np.array([(r["true"]["avg"] - r["tb"]) / r["span"] for r in recs])
    coef = np.polyfit(u_est, u_true, a.degree)

    def corrected(r):   # corrected temperatures inside the box (picture unit)
        return r["tb"] + np.polyval(coef, (r["inside"] - r["tb"]) / r["span"]) * r["span"]

    cor = [corrected(r) for r in recs]
    err_before = np.array([(r["inside"].mean() - r["true"]["avg"]) * to_c(r) for r in recs])
    err_avg = np.array([(c.mean() - r["true"]["avg"]) * to_c(r) for c, r in zip(cor, recs)])

    def best_q(grid, key):
        idx = [i for i, r in enumerate(recs) if key in r["true"]]
        if not idx: return grid[-1], None
        maes = [np.mean([abs(np.percentile(cor[i], q) - recs[i]["true"][key]) * to_c(recs[i]) for i in idx]) for q in grid]
        q = grid[int(np.argmin(maes))]
        err = np.array([(np.percentile(cor[i], q) - recs[i]["true"][key]) * to_c(recs[i]) for i in idx])
        return q, err

    q_max, err_max = best_q(QMAX, "max"); q_min, err_min = best_q(QMIN, "min")
    sd = lambda e: None if e is None else round(float(np.sqrt(np.mean(e ** 2))), 3)
    print("\n                         error in deg C (RMS)")
    print(f"  Average  before correction: {sd(err_before)}   after: {sd(err_avg)}")
    print(f"  Max  (uses percentile {q_max}):  {sd(err_max)}")
    print(f"  Min  (uses percentile {q_min}):  {sd(err_min)}")
    json.dump({"degree": a.degree, "coef": coef.tolist(), "q_max": q_max, "q_min": q_min,
               "unc_c": {"avg": sd(err_avg), "max": sd(err_max), "min": sd(err_min)},
               "n_images": len(recs)}, open("models/colorbar_calibration.json", "w"), indent=2)
    with open("models/learning_report.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["file", "unit", "true_avg", "est_avg_before", "est_avg_after", "true_max", "est_max", "true_min", "est_min"])
        for r, c in zip(recs, cor):
            w.writerow([r["name"], r["unit"], r["true"]["avg"], round(r["inside"].mean(), 2), round(c.mean(), 2),
                        r["true"].get("max"), round(float(np.percentile(c, q_max)), 2),
                        r["true"].get("min"), round(float(np.percentile(c, q_min)), 2)])
    print("Saved models/colorbar_calibration.json, models/learning_report.csv")
    print("Typical error you can expect in reports: about +-%s deg C on the hotspot." % sd(err_max))


if __name__ == "__main__":
    main()
