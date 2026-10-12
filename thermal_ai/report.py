"""PDF inspection report (reportlab)."""
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from .diagnosis import Finding

SEV_COLORS = {
    0: colors.HexColor("#2e9e44"),
    1: colors.HexColor("#d4a400"),
    2: colors.HexColor("#e67e00"),
    3: colors.HexColor("#c62828"),
}
SEV_LABEL = {0: "NORMAL", 1: "MONITOR", 2: "REPAIR SOON", 3: "MAJOR"}
DISCLAIMER = (
    "This report is an automated aid based on general thermography guidance (NETA / ISO 18434 style "
    "thresholds). It shows the condition at the time of the scan and is not a guarantee of future "
    "performance. Findings must be verified by a qualified thermographer / electrical engineer against "
    "the site standard and manufacturer limits. Temperatures marked PREDICTED are estimated from the picture's colours and carry the stated error."
)


def _png(img_bgr: np.ndarray) -> BytesIO:
    ok, buf = cv2.imencode(".png", img_bgr)
    return BytesIO(buf.tobytes())


def _image(img_bgr: np.ndarray, max_w: float, max_h: float, jpeg: bool = False) -> Image:
    h, w = img_bgr.shape[:2]
    s = min(max_w / w, max_h / h)
    if jpeg:                                            # photos: smaller file
        if w > 1000:
            img_bgr = cv2.resize(img_bgr, (1000, int(h * 1000 / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Image(BytesIO(buf.tobytes()), width=w * s, height=h * s)
    return Image(_png(img_bgr), width=w * s, height=h * s)


def _crop(display: np.ndarray, f: Finding, pad: int = 10) -> np.ndarray:
    h, w = display.shape[:2]
    x1, y1, x2, y2 = f.bbox
    c = display[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)].copy()
    hx, hy = f.hotspot_xy[0] - max(0, x1 - pad), f.hotspot_xy[1] - max(0, y1 - pad)
    cv2.drawMarker(c, (int(hx), int(hy)), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
    return c


def build_pdf(
    findings: List[Finding],
    display: np.ndarray,
    annotated: np.ndarray,
    meta: Dict[str, str],
    keys: Optional[Dict[int, str]] = None,
    trends: Optional[Dict[int, str]] = None,
    visible: Optional[np.ndarray] = None,
) -> bytes:
    """keys / trends are dicts keyed by id(finding)."""
    keys = {k: v.replace('|', ' / ') for k, v in (keys or {}).items()}
    trends = trends or {}
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=18 * mm,
                            title="Thermal Inspection Report", author="Thermal Fault AI")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["Normal"], fontSize=9, leading=12)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#444444"))
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, alignment=0, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4)
    W = A4[0] - 30 * mm

    story = [Paragraph("Thermal Inspection Report", h1),
             Paragraph("Electrical equipment infrared survey", small), Spacer(1, 6)]

    # ---- meta table
    rows = [[Paragraph(f"<b>{k}</b>", cell), Paragraph(str(v), cell)] for k, v in meta.items()]
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    right += [["", ""]] * (len(left) - len(right))
    mt = Table([l + r for l, r in zip(left, right)],
               colWidths=[32 * mm, W / 2 - 32 * mm, 32 * mm, W / 2 - 32 * mm])
    mt.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f0f0f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [mt, Spacer(1, 8)]

    # ---- overall summary
    worst = max((f.severity for f in findings), default=0)
    counts = {s: sum(1 for f in findings if f.severity == s) for s in range(4)}
    banner = Table([[Paragraph(
        f"<font color='white'><b>OVERALL: {SEV_LABEL[worst]}</b> &nbsp;|&nbsp; "
        f"{len(findings)} items &nbsp;|&nbsp; Major {counts[3]} &nbsp; Repair soon {counts[2]} &nbsp; "
        f"Monitor {counts[1]} &nbsp; Normal {counts[0]}</font>", body)]], colWidths=[W])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), SEV_COLORS[worst]),
                                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story += [banner, Spacer(1, 8)]
    if visible is not None:
        half = W / 2 - 3 * mm
        pair = Table([[_image(visible, half, 85 * mm, jpeg=True), _image(annotated, half, 85 * mm)],
                      [Paragraph("<b>Original (visible-light) photo</b>", small),
                       Paragraph("<b>Thermal image with hotspots</b> - numbers match the # column; box colour = severity; "
                                 "white cross = hottest point", small)]],
                     colWidths=[W / 2, W / 2])
        pair.setStyle(TableStyle([("VALIGN", (0, 0), (-1, 0), "MIDDLE"), ("VALIGN", (0, 1), (-1, 1), "TOP"),
                                  ("ALIGN", (0, 0), (-1, 0), "CENTER"), ("BOX", (0, 0), (0, 0), 0.3, colors.HexColor("#cccccc")),
                                  ("BOX", (1, 0), (1, 0), 0.3, colors.HexColor("#cccccc"))]))
        story += [pair, Paragraph("The visible and thermal cameras have different fields of view, so the two pictures "
                                  "do not line up exactly.", small)]
    else:
        story += [_image(annotated, W, 95 * mm),
                  Paragraph("Annotated thermal image. Numbers match the # column below; box colour = severity; "
                            "white cross = hottest point.", small)]

    # ---- summary table
    story.append(Paragraph("Summary of findings", h2))
    ordered = sorted(findings, key=lambda f: (-f.severity, -f.max_temp))
    data = [[Paragraph(f"<b>{h}</b>", cell) for h in
             ["#", "Equipment", "Max temp", "Rise vs similar", "Rise vs ambient", "Severity", "Likely fault"]]]
    for i, f in enumerate(ordered, 1):
        sim = "n/a" if f.delta_similar is None else f"{f.delta_similar:+.1f} C"
        data.append([str(i), Paragraph(keys.get(id(f), f.equipment), cell), f"{f.max_temp:.1f} C", sim,
                     f"{f.delta_ambient:+.1f} C",
                     Paragraph(f"<font color='white'><b>{SEV_LABEL[f.severity]}</b></font>", cell),
                     Paragraph(f.fault, cell)])
    st = Table(data, colWidths=[7 * mm, 33 * mm, 18 * mm, 22 * mm, 22 * mm, 24 * mm, W - 126 * mm], repeatRows=1)
    style = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
             ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    for i, f in enumerate(ordered, 1):
        style.append(("BACKGROUND", (5, i), (5, i), SEV_COLORS[f.severity]))
    st.setStyle(TableStyle(style))
    story.append(st)

    # ---- detail per finding (those needing attention first)
    story.append(Paragraph("Detailed findings", h2))
    for i, f in enumerate(ordered, 1):
        title = Paragraph(
            f"<b>{i}. {keys.get(id(f), f.equipment)}</b> &nbsp; "
            f"<font color='{SEV_COLORS[f.severity].hexval().replace('0x', '#')}'><b>{f.severity_name}</b></font>", body)
        lines = [f"<b>Max temperature:</b> {f.max_temp:.1f} C at pixel {f.hotspot_xy}"
                 + (f" ({f.temp_source})" if f.temp_source else "") + f"; "
                 f"hot area {f.spread * 100:.0f}% of equipment box",
                 f"<b>Likely fault:</b> {f.fault}",
                 f"<b>Risk:</b> {f.risk}",
                 f"<b>Recommended action:</b> {f.action}",
                 f"<b>Basis:</b> {'; '.join(f.reasons)}"]
        if id(f) in trends:
            lines.append(f"<b>Trend:</b> {trends[id(f)]}")
        text = [Paragraph(l, body) for l in lines]
        if f.severity >= 1:
            thumb = _image(_crop(display, f), 45 * mm, 38 * mm)
            t = Table([[thumb, text]], colWidths=[48 * mm, W - 48 * mm])
        else:
            t = Table([[text]], colWidths=[W])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                               ("LINEBEFORE", (0, 0), (0, -1), 3, SEV_COLORS[f.severity])]))
        story.append(KeepTogether([title, Spacer(1, 2), t, Spacer(1, 6)]))

    # ---- guidance + disclaimer
    story.append(Paragraph("Severity criteria used", h2))
    crit = Table([
        ["Level", "Rise vs similar component", "Rise vs ambient"],
        ["Monitor (possible deficiency)", "1 - 3 C", "1 - 10 C"],
        ["Repair soon (probable deficiency)", "4 - 15 C", "11 - 20 C"],
        ["Major (repair immediately)", "> 15 C", "> 20 C"],
    ], colWidths=[W * 0.4, W * 0.3, W * 0.3])
    crit.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
                              ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
                              ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story += [crit, Spacer(1, 4),
              Paragraph("Absolute temperature limits per equipment type are also applied (configurable).", small),
              Spacer(1, 8), Paragraph("<b>Disclaimer.</b> " + DISCLAIMER, small)]

    def footer(c, d):
        c.saveState()
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.grey)
        c.drawString(15 * mm, 10 * mm, f"Thermal Fault AI  |  {meta.get('Panel / location', '')}  |  "
                                       f"generated {datetime.now():%Y-%m-%d %H:%M}")
        c.drawRightString(A4[0] - 15 * mm, 10 * mm, f"Page {d.page}")
        c.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
