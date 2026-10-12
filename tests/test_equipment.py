import sys, os, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, cv2
from thermal_ai import detector, pipeline
from thermal_ai.detector import Detection, to_yolo_lines
from thermal_ai.temperature import ThermalData, colorize

# synthetic 3-phase panel (real temperatures): phase 2 has a hot terminal
t = np.full((240, 320), 30.0, np.float32)
for cx, d in [(80, 6), (160, 38), (240, 7)]:
    t[85:95, cx - 4:cx + 4] += d
    t[70:170, cx - 20:cx + 20] += 3
data = ThermalData(t, colorize(t), True, "demo", source="csv")
boxes = [Detection("terminal_block", (cx - 25, 60, cx + 25, 180), 0.9) for cx in (240, 80, 160)]   # unsorted on purpose

# 1) equipment mode: worst item is the middle block, labels run left -> right
r = pipeline.analyze_loaded(data, 30, dets=boxes)
assert len(r.findings) == 3
names = [r.labels[id(f)] for f in r.findings]
assert names == ["terminal_block 1", "terminal_block 2", "terminal_block 3"], names
worst = max(r.findings, key=lambda f: f.severity)
assert r.labels[id(worst)] == "terminal_block 2" and worst.severity == 3
assert "terminal" in worst.fault.lower() and "not identified" not in worst.fault
assert all(f.severity < 3 for f in r.findings if f is not worst)          # phase comparison works

# 2) hottest point outside every box -> extra 'unknown' finding
t2 = t.copy(); t2[200:206, 20:26] += 50
d2 = ThermalData(t2, colorize(t2), True, "demo", source="csv")
r2 = pipeline.analyze_loaded(d2, 30, dets=boxes)
assert len(r2.findings) == 4 and "outside" in list(r2.labels.values())[-1]

# 3) no boxes -> old 'unknown' behaviour still works
r3 = pipeline.analyze_loaded(data, 30)
assert all(f.equipment == "unknown" for f in r3.findings)

# 4) predicted source: box containing the camera's Max uses the measured number
dp = ThermalData(t, colorize(t), False, "demo", source="colorbar", onscreen={"max": 71.5, "min": 30.0}, uncertainty_c=0.6)
rp = pipeline.analyze_loaded(dp, 30, dets=boxes)
mid = [f for f in rp.findings if r.labels.get(id(f)) is None and f.bbox[0] == 135][0]
assert abs(mid.max_temp - 71.5) < 0.01 and "MEASURED by the camera" in mid.temp_source
others = [f for f in rp.findings if f is not mid]
assert all("PREDICTED" in f.temp_source for f in others)

# 5) detect_equipment never raises: missing model -> clear message
dets, msg = pipeline.detect_equipment(data, weights="models/does_not_exist.pt")
assert dets == [] and "No trained model" in msg

# 6) YOLO stand-in wired through analyze_image
class FakeDet:
    conf = 0.25
    def detect(self, img): return boxes
detector.get_detector = lambda w, c=0.25: (FakeDet(), "")
pipeline.get_detector = detector.get_detector
csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False); np.savetxt(csv.name, t, delimiter=","); csv.close()
r6 = pipeline.analyze_image(csv.name, 30, use_yolo=True)
assert len(r6.findings) == 3 and any(f.equipment == "terminal_block" for f in r6.findings)

# 7) YOLO label format
line = to_yolo_lines([Detection("MCB", (0, 0, 100, 50))], 200, 100)[0]
assert line.split()[0] == "4" and abs(float(line.split()[1]) - 0.25) < 1e-6 and abs(float(line.split()[3]) - 0.5) < 1e-6
print("PASS equipment (YOLO) integration tests")
