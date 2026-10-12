"""Create a synthetic thermal CSV (3-phase terminal blocks, phase B has a loose connection)
so you can test the app without a camera.   python make_demo.py"""
import numpy as np
from pathlib import Path

H, W = 240, 320
rng = np.random.default_rng(0)
t = 30 + rng.normal(0, 0.3, (H, W)).astype(np.float32)
yy, xx = np.mgrid[0:H, 0:W]

def blob(cx, cy, amp, sx, sy):
    return amp * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))

for cx, amp in [(80, 8), (160, 38), (240, 9)]:       # R, Y(B), B phases
    t += blob(cx, 120, 4, 25, 40)                   # normal body warmth
    t += blob(cx, 90, amp, 6, 6)                    # terminal hotspot
Path("data").mkdir(exist_ok=True)
np.savetxt("data/demo_panel.csv", t, delimiter=",", fmt="%.2f")
print("saved data/demo_panel.csv  (boxes: x 55-105, 135-185, 215-265, y 60-180)")
