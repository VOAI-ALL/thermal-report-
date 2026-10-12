# Thermal Hotspot AI

## PHASE 1 (this version): temperature -> hotspot -> report
Upload a thermal picture and the project will:
1. **Use the real temperature if it is available** (CSV, radiometric file, or the camera's own Max/Avg/Min numbers printed on the picture)
2. **Otherwise PREDICT the temperature** from the picture's colour scale bar, using what the AI learned from your pictures
3. **Find the hotspot** (position + temperature) and rate it (monitor / repair soon / major)
4. **Create a PDF report** (also CSV) - with the **original (visible) photo next to the thermal image** when one is available:
   taken automatically from Fluke `.is2` and FLIR radiometric files, or upload your own in the app ("Original photo - optional")

Equipment identification (CT, MCB ...) with **YOLO** is built in - see "Equipment identification (YOLO)" below. It works with or without a trained model.

## Setup (once)
```bash
conda create -n thermal python=3.11 -y
conda activate thermal
pip install -r requirements.txt
conda install -c conda-forge tesseract -y      # OCR: reads the scale numbers + camera numbers
```

**Important:** `pip install pytesseract` only installs the Python wrapper. The Tesseract PROGRAM must be installed
separately (the conda command above does that). Check with `tesseract --version`. After installing, restart `streamlit run app.py`.

## Run
```bash
streamlit run app.py          # upload data/sample_fluke_ir.jpg to try
```

## Teach the AI your camera's pattern (do this with your pictures)
Put your Fluke pictures (colour bar + Max/Avg/Min box visible) in `data/calibration/`, then:
```bash
python learn_temperature.py data/calibration
```
It reads the camera's own numbers as ground truth and learns:
* a correction so colour-based temperatures are not biased (the sample was ~0.8 C too cold before learning)
* which part of the hot area best matches the camera's Max / Min
* the typical error (shown in every report)
* the colour palette (lets pictures WITHOUT a bar be read if you type min/max)

Saved in `models/`. **Run it again whenever you add more pictures** - the AI improves with data.
Read the printed summary: the "error in deg C" lines are your real accuracy.
If OCR misreads a number, add `labels.csv` (filename,max,avg,min) to the folder.

## Fluke .is2 files
Upload a `.is2` file directly. Two cases (the app chooses automatically):
1. **Older / supported cameras** (e.g. Ti300, Ti480P): exact per-pixel temperatures through `fluke-thermal-reader`.
2. **Newer cameras (e.g. TiS55+)**: these files have NO per-pixel temperature array, so the built-in reader (`thermal_ai/is2.py`) uses
   what the file does contain: the camera's **Hot point position**, the **Max / Min / centre numbers** printed on its screen picture
   (read with OCR) and its exact colour palette. The hotspot temperature and position are therefore the CAMERA'S OWN values;
   the rest of the heat map is only predicted (rough, about +-0.5 C, checked on one file) and is a guide, not a measurement.
   Screen layout constants are for the TiS55+; other newer models may need new constants (send me a file).
If neither works the app says so - export a **CSV** from Fluke SmartView instead (always works).
`.is3` (video) is not supported.

## Equipment identification (YOLO)
In the app: sidebar "Equipment (YOLO)" -> the model draws a box + class for each item (CT, VT, busbar, switchgear, MCB, MCCB,
cable, terminal_block). The **Equipment table** shows them: you can fix a class, move a box, delete rows or **add boxes by hand
(works even with no trained model)**. Every item then gets its own hotspot, severity and cause; items of the same class are
compared with each other (phases R/Y/B), which is what finds a single bad connection. Leave the table empty = plain hotspots.

**Training the model (do this once you have pictures):**
```bash
pip install ultralytics
python prepare_images.py data/calibration     # crops colour bar / takes the heat-map part of .is2 -> data/to_label
# label the pictures: Roboflow or CVAT (YOLOv8 export)  - OR use the app: correct the table, click
#   "Save this picture + boxes for YOLO training" (writes image + label file to data/to_label)
python split_dataset.py data/to_label         # or path of the Roboflow export -> data/dataset (80/20)
python train.py                               # trains, copies best model to models/best.pt
```
Class order is fixed: 0 CT, 1 VT, 2 busbar, 3 switchgear, 4 MCB, 5 MCCB, 6 cable, 7 terminal_block.
Tip: labelling in the app is fastest - the model suggests boxes, you correct them, retrain, repeat (it improves each round).
Train on the same kind of picture you use (camera screen picture / `.is2` heat map). CSV files are shown to YOLO as a
false-colour image, which a model trained on camera pictures may not recognise well.

## How a picture is handled (priority)
| Picture has | Temperature | Label in report |
|---|---|---|
| CSV / FLIR radiometric / older Fluke .is2 | real | MEASURED |
| Newer Fluke .is2 (TiS55+) | camera's Max + hot point; rough heat map | MEASURED by camera |
| Colour bar + camera's Max number | hotspot = camera's number, map predicted | MEASURED by camera |
| Colour bar only | predicted from bar + learned correction | PREDICTED (+- error) |
| No bar | learned palette + min/max you type | PREDICTED (less reliable) |
| Nothing | brightness guess | ROUGH |

## Files
* `app.py` - Phase 1 screen | `thermal_ai/pipeline.py` - the whole flow
* `thermal_ai/temperature.py` - chooses the temperature source | `colorbar.py` - colour-bar reading
* `thermal_ai/onscreen.py` - reads the camera's box + Max/Avg/Min | `diagnosis.py` - severity rules
* `thermal_ai/report.py` - PDF | `learn_temperature.py` - the learning step
* YOLO: `thermal_ai/detector.py`, `prepare_images.py`, `split_dataset.py`, `train.py`
* Tests: `python tests/test_pipeline.py`, `test_colorbar.py`, `test_diagnosis.py`, `test_is2.py`, `test_equipment.py`, `test_ocr_missing.py`

## Limits
* Predicted temperatures are estimates; real inspections should use CSV / radiometric files when possible.
* Severity thresholds are general guidance (NETA style) - set your own limits in `thermal_ai/config.py`.
* One picture shows the current condition. Trend prediction needs repeated scans (`thermal_ai/history.py`).
