import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from thermal_ai import ocr
from thermal_ai.pipeline import analyze_image
here = os.path.dirname(__file__)
ocr.find_tesseract = lambda: None            # pretend Tesseract is not installed
for f in ("sample_fluke_tis55.is2", "sample_fluke_ir.jpg"):
    try:
        analyze_image(os.path.join(here, "..", "data", f), 30); raise SystemExit("should have failed")
    except ocr.TesseractMissing as e:
        assert "conda install -c conda-forge tesseract" in str(e)
print("PASS missing-tesseract gives a clear message")
