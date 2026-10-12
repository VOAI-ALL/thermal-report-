"""Rule-based thermal diagnosis (NETA / ISO 18434 style).

Severity uses the WORST of:
  * delta vs similar components (e.g. R/Y/B phases)     1-3 / 4-15 / >15 C
  * delta vs ambient                                     1-10 / 11-20 / >20 C
  * absolute temperature limit for that equipment class
The fault type is inferred from hotspot SHAPE (small vs wide) and POSITION (edge vs centre).
"""
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config as C
from .detector import Detection


@dataclass
class Finding:
    equipment: str
    bbox: Tuple[int, int, int, int]
    max_temp: float
    mean_temp: float
    hotspot_xy: Tuple[int, int]      # in full-image coordinates
    delta_similar: Optional[float]
    delta_ambient: float
    delta_used: float                # delta (load-corrected if load given) used for severity
    spread: float                    # fraction of box that is hot
    severity: int
    severity_name: str
    fault: str
    action: str
    risk: str
    reasons: List[str]
    temp_source: str = ""

    def to_dict(self):
        return asdict(self)


def _sev_similar(d: float) -> int:
    return 0 if d < 1 else 1 if d <= 3 else 2 if d <= 15 else 3


def _sev_ambient(d: float) -> int:
    return 0 if d < 1 else 1 if d <= 10 else 2 if d <= 20 else 3


def _load_factor(load_pct: Optional[float]) -> float:
    if load_pct is None or load_pct < C.MIN_LOAD_PCT:
        return 1.0
    return min((100.0 / load_pct) ** 2, C.MAX_LOAD_FACTOR)


_LOCAL = {
    "terminal_block": "Loose or oxidised screw terminal (high-resistance joint)",
    "busbar": "Loose or corroded busbar joint / bolted connection",
    "switchgear": "Loose connection or contact degradation inside the compartment",
    "MCB": "Loose terminal or lug at the breaker",
    "MCCB": "Loose terminal or lug at the breaker",
    "cable": "Loose or damaged cable lug / termination",
    "CT": "Loose or corroded terminal connection on the CT",
    "VT": "Loose or corroded terminal connection on the VT",
}
_LOCAL_CENTRE = {
    "CT": "Possible internal winding / insulation degradation or overloaded burden (CT body hotspot)",
    "VT": "Possible internal winding / insulation degradation (VT body hotspot)",
    "MCB": "Worn or oxidised internal contacts / overheating inside breaker",
    "MCCB": "Worn or oxidised internal contacts / overheating inside breaker",
    "cable": "Localised insulation damage or broken conductor strands",
}


def _fault(cls: str, spread: float, edge: bool, has_peers: bool, d_sim: Optional[float]) -> str:
    if cls == "unknown":
        if spread < C.LOCALIZED_FRACTION:
            return "Localised hotspot (possible loose / high-resistance connection) - equipment not identified yet"
        if spread >= C.WIDESPREAD_FRACTION:
            return "Widespread warm area (possible overload / poor ventilation) - equipment not identified yet"
        return "Warm area found (possible developing connection problem or load) - equipment not identified yet"
    if spread < C.LOCALIZED_FRACTION:
        if not edge and cls in _LOCAL_CENTRE:
            return _LOCAL_CENTRE[cls]
        return _LOCAL.get(cls, "Localised high-resistance connection (check terminals and lugs)")
    if spread >= C.WIDESPREAD_FRACTION:
        if has_peers and d_sim is not None and d_sim >= 1:
            return "Overloaded or unbalanced phase (this phase carries more current than its peers)"
        return "General overload or poor ventilation / cooling (all similar parts equally warm)"
    return "Developing high-resistance connection or partial overload - inspect connections and load"


_ACTION = {
    0: "No action. Keep in routine inspection schedule.",
    1: "Monitor: re-scan at next inspection or within 1-3 months; check load.",
    2: "Schedule repair soon: de-energise safely, inspect and re-torque / clean connections.",
    3: "Immediate action: reduce load / isolate and repair as soon as safely possible.",
}
_RISK = {
    0: "Low",
    1: "Low - early sign only",
    2: "Medium - likely to degrade if left alone",
    3: "High - failure, arcing or fire possible in the near term",
}


def analyze(
    detections: List[Detection],
    temp: np.ndarray,
    ambient: float,
    load_pct: Optional[float] = None,
    tmax_override: Optional[List[Optional[float]]] = None,
) -> List[Finding]:
    h, w = temp.shape
    stats = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            stats.append(None)
            continue
        crop = temp[y1:y2, x1:x2]
        iy, ix = np.unravel_index(np.argmax(crop), crop.shape)
        tmax, med = float(crop.max()), float(np.median(crop))
        if tmax_override and tmax_override[len(stats)] is not None:
            tmax = float(tmax_override[len(stats)])
        rng = tmax - med
        spread = float((crop > med + 0.5 * rng).mean()) if rng > 0.5 else 0.0
        stats.append(dict(
            tmax=tmax, mean=float(crop.mean()), xy=(x1 + int(ix), y1 + int(iy)),
            rx=ix / max(crop.shape[1] - 1, 1), ry=iy / max(crop.shape[0] - 1, 1), spread=spread,
        ))

    factor = _load_factor(load_pct)
    findings: List[Finding] = []
    for i, (d, s) in enumerate(zip(detections, stats)):
        if s is None:
            continue
        peers = [stats[j]["tmax"] for j, o in enumerate(detections)
                 if j != i and o.cls == d.cls and d.cls != "unknown" and stats[j] is not None]
        d_sim = (s["tmax"] - float(np.median(peers))) if peers else None
        d_amb = s["tmax"] - ambient

        sevs, reasons = [], []
        if d_sim is not None:
            v = d_sim * factor
            sevs.append(_sev_similar(v))
            reasons.append(f"+{d_sim:.1f} C vs similar {d.cls} (x{factor:.2f} load correction)")
        v_amb = d_amb * factor
        s_amb = _sev_ambient(v_amb)
        if d_sim is not None and _sev_similar(d_sim * factor) == 0:
            # not an outlier vs its peers: ambient rise alone means overload/warm, not a bad joint
            s_amb = min(s_amb, 1 if v_amb <= 20 else 2)
        sevs.append(s_amb)
        reasons.append(f"+{d_amb:.1f} C vs ambient")

        caution, critical = C.ABS_LIMITS.get(d.cls, C.ABS_LIMITS["unknown"])
        if s["tmax"] >= critical:
            sevs.append(3); reasons.append(f"{s['tmax']:.0f} C >= critical limit {critical} C")
        elif s["tmax"] >= caution:
            sevs.append(2); reasons.append(f"{s['tmax']:.0f} C >= caution limit {caution} C")

        sev = max(sevs)
        edge = s["rx"] < 0.2 or s["rx"] > 0.8 or s["ry"] < 0.2 or s["ry"] > 0.8
        fault = "No thermal anomaly" if sev == 0 else _fault(d.cls, s["spread"], edge, bool(peers), d_sim)

        findings.append(Finding(
            equipment=d.cls, bbox=d.bbox, max_temp=s["tmax"], mean_temp=s["mean"],
            hotspot_xy=s["xy"], delta_similar=d_sim, delta_ambient=d_amb,
            delta_used=(d_sim * factor if d_sim is not None else d_amb * factor),
            spread=s["spread"], severity=sev, severity_name=C.SEVERITY_NAMES[sev],
            fault=fault, action=_ACTION[sev], risk=_RISK[sev], reasons=reasons,
        ))
    return findings
