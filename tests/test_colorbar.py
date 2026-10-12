import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import shutil, cv2
if not shutil.which("tesseract"):
    print("SKIP: tesseract not installed"); sys.exit(0)
from thermal_ai.colorbar import read_colorbar, estimate_temperature
from thermal_ai.onscreen import find_box, read_stats

img = cv2.imread(os.path.join(os.path.dirname(__file__), "..", "data", "sample_fluke_ir.jpg"))
bar = read_colorbar(img)
assert bar.unit == "F" and bar.n_labels >= 15
assert abs(bar.t_top - 94.1) < 0.1 and abs(bar.t_bot - 89.9) < 0.1, (bar.t_top, bar.t_bot)
t, ov, xe = estimate_temperature(img, bar)
assert 89.8 <= t[:, :xe].min() and t[:, :xe].max() <= 94.2        # stays inside the bar range
assert find_box(img, xe) == (80, 60, 240, 180)
assert read_stats(img, xe) == {"max": 93.7, "avg": 92.7, "min": 91.4}
print("PASS colour-bar tests")
