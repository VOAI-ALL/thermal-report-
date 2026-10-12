"""streamlit run app.py
Upload a thermal picture -> temperature (read or predicted) -> [equipment boxes] -> hotspot -> PDF report."""
import os
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from thermal_ai.config import CLASSES
from thermal_ai.detector import Detection, to_yolo_lines
from thermal_ai.pipeline import analyze_loaded, detect_equipment, load_data, make_pdf
from thermal_ai.temperature import REAL_SOURCES

st.set_page_config(page_title="Thermal Hotspot AI", layout="wide")
st.title("Thermal Hotspot AI")

with st.sidebar:
    st.header("Inputs")
    up = st.file_uploader("Thermal picture (Fluke .is2, JPG/PNG, FLIR JPG or CSV)", type=["jpg", "jpeg", "png", "csv", "is2"])
    vis_up = st.file_uploader("Original (visible) photo - optional", type=["jpg", "jpeg", "png"])
    panel = st.text_input("Panel / location", "PANEL-01")
    inspector = st.text_input("Inspector", "")
    camera = st.text_input("Camera", "Fluke")
    ambient = st.number_input("Ambient temperature (C)", value=30.0)
    use_load = st.checkbox("I know the load at the time of the scan")
    load_pct = st.number_input("Load (% of rated)", 1.0, 150.0, 60.0) if use_load else None
    st.header("Equipment (YOLO)")
    use_yolo = st.checkbox("Identify equipment automatically", value=True)
    weights = st.text_input("Model file", "models/best.pt")
    conf = st.slider("Detection confidence", 0.05, 0.9, 0.25)
    with st.expander("Advanced"):
        unit_hint = st.selectbox("Scale unit if not detected", ["F", "C"])
        manual = st.checkbox("Type the scale myself (picture has no readable colour bar)")
        t_min = st.number_input("Scale min (in the picture's unit)", value=80.0) if manual else None
        t_max = st.number_input("Scale max (in the picture's unit)", value=100.0) if manual else None

if not up:
    st.info("Upload a thermal picture to start. Try data/sample_fluke_ir.jpg or data/sample_fluke_tis55.is2")
    st.stop()


def _save(upload) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload.name)[1]) as t:
        t.write(upload.getvalue())
    return t.name


# ---- stage 1: load (cached so the app is fast when you edit boxes)
key = (up.name, up.size, vis_up.name if vis_up else None, t_min, t_max, unit_hint)
if st.session_state.get("key") != key:
    try:
        st.session_state.data = load_data(_save(up), t_min, t_max, unit_hint, _save(vis_up) if vis_up else None)
    except Exception as e:
        st.error(str(e)); st.stop()
    st.session_state.key = key
    st.session_state.pop("det_key", None)
data = st.session_state.data
(st.success if data.source in REAL_SOURCES else st.warning)(data.note)

# ---- stage 2: equipment boxes (YOLO, then editable by you)
dkey = (key, use_yolo, weights, conf)
if st.session_state.get("det_key") != dkey:
    boxes, msg = detect_equipment(data, weights, conf) if use_yolo else ([], "Automatic detection is off.")
    st.session_state.boxes, st.session_state.det_msg, st.session_state.det_key = boxes, msg, dkey
st.subheader("Equipment")
st.caption(st.session_state.det_msg + "  You can edit the table, delete rows or add rows by hand "
           "(leave it empty to get plain hotspots).")
table = pd.DataFrame([{"cls": b.cls, "x1": b.bbox[0], "y1": b.bbox[1], "x2": b.bbox[2], "y2": b.bbox[3]}
                      for b in st.session_state.boxes], columns=["cls", "x1", "y1", "x2", "y2"])
table = st.data_editor(table, num_rows="dynamic", use_container_width=True, key=f"tbl{hash(dkey)}",
                       column_config={"cls": st.column_config.SelectboxColumn("cls", options=CLASSES)})
dets = [Detection(r.cls, (int(r.x1), int(r.y1), int(r.x2), int(r.y2)))
        for r in table.dropna().itertuples() if r.cls in CLASSES]

# ---- stage 3: analyse
res = analyze_loaded(data, ambient, load_pct, dets or None)
top = max(res.findings, key=lambda f: (f.severity, f.max_temp))
c1, c2, c3 = st.columns(3)
c1.metric("Hottest item", f"{top.max_temp:.1f} °C", help=res.labels[id(top)])
c2.metric("Rise over ambient", f"{top.delta_ambient:+.1f} °C")
c3.metric("Worst severity", top.severity_name.split(" - ")[0])
st.write(f"**Hottest item:** {res.labels[id(top)]}  |  **Temperature source:** {top.temp_source}")
if res.uncertainty_c and "PREDICTED" in top.temp_source:
    st.write(f"Predicted values can be off by about ±{res.uncertainty_c:.1f} °C.")

left, right = st.columns([3, 2])
with left:
    if res.visible is not None:
        a, b = st.columns(2)
        a.image(cv2.cvtColor(res.visible, cv2.COLOR_BGR2RGB), caption="Original (visible) photo", use_container_width=True)
        b.image(cv2.cvtColor(res.annotated, cv2.COLOR_BGR2RGB), caption="Thermal with hotspots", use_container_width=True)
    else:
        st.image(cv2.cvtColor(res.annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.caption(f"Scene: min {res.scene['min']:.1f}, mean {res.scene['mean']:.1f}, max {res.scene['max']:.1f} °C")
with right:
    icons = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴"}
    for f in sorted(res.findings, key=lambda f: (-f.severity, -f.max_temp)):
        with st.expander(f"{icons[f.severity]} {res.labels[id(f)]} - {f.max_temp:.1f} °C", expanded=(f is top)):
            st.write(f"**Severity:** {f.severity_name}")
            st.write(f"**Likely cause:** {f.fault}")
            st.write(f"**Action:** {f.action}")
            st.write(f"**Why:** " + "; ".join(f.reasons))
            st.write(f"**Source:** {f.temp_source}")

# ---- outputs
meta = {"Panel / location": panel, "Inspector": inspector or "-", "Camera": camera or "-",
        "Ambient": f"{ambient:.1f} C", "Load at scan": f"{load_pct:.0f} % of rated" if load_pct else "not recorded",
        "File": up.name}
st.download_button("Download PDF report", make_pdf(res, meta), f"thermal_report_{panel}.pdf",
                   "application/pdf", type="primary")
rows = [dict(item=res.labels[id(f)], temp_c=round(f.max_temp, 2), x=f.hotspot_xy[0], y=f.hotspot_xy[1],
             severity=f.severity_name, cause=f.fault, source=f.temp_source) for f in res.findings]
st.download_button("Download CSV", pd.DataFrame(rows).to_csv(index=False), "hotspots.csv")

# ---- every corrected picture becomes training data
st.divider()
if st.button("Save this picture + boxes for YOLO training"):
    out = Path("data/to_label"); out.mkdir(parents=True, exist_ok=True)
    stem = Path(up.name).stem
    cv2.imwrite(str(out / f"{stem}.jpg"), data.display)
    h, w = data.display.shape[:2]
    lines = to_yolo_lines(dets, w, h)
    (out / f"{stem}.txt").write_text("\n".join(lines))
    st.success(f"Saved to data/to_label/{stem}.jpg + .txt ({len(lines)} boxes). "
               "When you have enough: python split_dataset.py data/to_label, then python train.py")
