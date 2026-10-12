"""Find the Tesseract OCR program (it is a separate program, not just a Python package)."""
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

INSTALL_HELP = (
    "Tesseract OCR is not installed (or not found). It is needed to read the numbers printed on thermal pictures.\n"
    "FIX (conda):  conda activate thermal   then   conda install -c conda-forge tesseract -y   then restart the app.\n"
    "Or install it from https://github.com/UB-Mannheim/tesseract/wiki (Windows) / 'brew install tesseract' (Mac) / "
    "'sudo apt install tesseract-ocr' (Linux).\n"
    "Check with:  tesseract --version"
)


class TesseractMissing(RuntimeError):
    pass


def find_tesseract() -> Optional[str]:
    p = shutil.which("tesseract")
    if p:
        return p
    prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
    candidates = [
        Path(prefix) / "Library" / "bin" / "tesseract.exe",     # conda on Windows
        Path(prefix) / "Scripts" / "tesseract.exe",
        Path(prefix) / "bin" / "tesseract",                     # conda on Mac / Linux
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path("/opt/homebrew/bin/tesseract"), Path("/usr/local/bin/tesseract"), Path("/usr/bin/tesseract"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def get_pytesseract():
    """Return the pytesseract module, pointed at the Tesseract program - or raise TesseractMissing."""
    try:
        import pytesseract
    except ImportError:
        raise TesseractMissing("Python package pytesseract is missing: pip install pytesseract\n" + INSTALL_HELP)
    exe = find_tesseract()
    if exe is None:
        raise TesseractMissing(INSTALL_HELP)
    pytesseract.pytesseract.tesseract_cmd = exe
    return pytesseract
