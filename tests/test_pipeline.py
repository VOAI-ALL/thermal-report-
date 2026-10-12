import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
if not shutil.which("tesseract"):
    print("SKIP: tesseract not installed"); sys.exit(0)
from thermal_ai.pipeline import analyze_image, make_pdf

r = analyze_image(os.path.join(os.path.dirname(__file__), "..", "data", "sample_fluke_ir.jpg"), 30)
top = r.findings[0]
assert abs(top.max_temp - 34.28) < 0.05 and "MEASURED" in top.temp_source   # 93.7 F read from the picture
assert 80 < top.hotspot_xy[0] < 240 and 60 < top.hotspot_xy[1] < 180         # inside the camera box
assert make_pdf(r, {"Panel / location": "T"})[:4] == b"%PDF"
print("PASS pipeline test")
