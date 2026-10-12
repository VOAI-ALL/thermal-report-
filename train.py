"""Train the equipment detector.   python train.py          (run from the project folder)"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
ds = root / "data" / "dataset"
if not list((ds / "images" / "train").glob("*")):
    sys.exit("No training images in data/dataset/images/train.\n"
             "Do: prepare_images.py -> label -> split_dataset.py first.")

# write a yaml with an absolute path (avoids path problems)
names = ["CT", "VT", "busbar", "switchgear", "MCB", "MCCB", "cable", "terminal_block"]
yaml = root / "data_abs.yaml"
yaml.write_text(f"path: {ds.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" +
                "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))

from ultralytics import YOLO
model = YOLO("yolov8n.pt")              # small + fast; good for a small dataset
model.train(
    data=str(yaml), epochs=100, imgsz=640, batch=8, patience=30,
    hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,    # keep colour changes mild for thermal palettes
    fliplr=0.5, mosaic=1.0, project=str(root / "runs"), name="equipment", exist_ok=True,
)
best = root / "runs" / "equipment" / "weights" / "best.pt"
if best.exists():
    (root / "models").mkdir(exist_ok=True)
    import shutil; shutil.copy(best, root / "models" / "best.pt")
    print("\nDONE. Model copied to models/best.pt  -> run: streamlit run app.py")
