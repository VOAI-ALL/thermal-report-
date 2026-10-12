"""Tests the .is2 integration with a STAND-IN reader (the real package needs internet to install).
It checks our code around it. It does NOT prove the real package reads your camera's files."""
import sys, os, types
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np, tempfile

# fake fluke_thermal_reader returning a 240x320 scene with one hot spot at (row 100, col 200) = 71.5 C
t = np.full((240, 320), 30.0, np.float32); t[98:103, 198:203] = 71.5
fake = types.ModuleType("fluke_thermal_reader")
fake.read_is2 = lambda p: {"data": t, "CameraModel": "Fluke TiX-demo", "BackgroundTemp": 25.0}
sys.modules["fluke_thermal_reader"] = fake

from thermal_ai.pipeline import analyze_image, make_pdf
f = tempfile.NamedTemporaryFile(suffix=".is2", delete=False); f.write(b"dummy"); f.close()
r = analyze_image(f.name, ambient_c=30)
top = r.findings[0]
assert r.data.source == "is2" and r.data.radiometric
assert abs(top.max_temp - 71.5) < 0.01 and "MEASURED" in top.temp_source
assert abs(top.hotspot_xy[0] - 200) <= 3 and abs(top.hotspot_xy[1] - 100) <= 3
assert top.severity == 3
assert make_pdf(r, {"Panel / location": "T"})[:4] == b"%PDF"

# error paths
sys.modules["fluke_thermal_reader"] = None   # simulate package not installed -> built-in reader -> dummy file must fail clearly
try:
    analyze_image(f.name, 30); raise SystemExit("should have failed")
except ValueError as e:
    assert "Export it as CSV" in str(e)
print("PASS is2 integration test (with stand-in reader)")

# ---- REAL file from a Fluke TiS55+ (new format, no per-pixel array)
import shutil
if shutil.which("tesseract"):
    sys.modules["fluke_thermal_reader"] = None
    r = analyze_image(os.path.join(os.path.dirname(__file__), "..", "data", "sample_fluke_tis55.is2"), 30)
    top = r.findings[0]
    assert r.data.source == "is2_screen" and r.data.onscreen == {"max": 39.3, "min": 34.7, "centre": 36.6}
    assert abs(top.max_temp - 39.3) < 0.01 and top.hotspot_xy == (162, 2)
    assert abs(r.scene["max"] - 39.3) < 0.01
    assert r.visible is not None and r.visible.shape == (960, 1280, 3)      # original photo found inside the .is2
    assert make_pdf(r, {"Panel / location": "T"})[:4] == b"%PDF"
    print("PASS real TiS55+ .is2 file")
else:
    print("SKIP real .is2 test: tesseract not installed")
