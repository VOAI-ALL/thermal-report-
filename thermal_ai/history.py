"""Inspection history + simple trend prediction (linear fit of delta-T over time)."""
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

COLUMNS = ["timestamp", "equipment_key", "max_temp", "delta_used", "severity", "load_pct"]


def append(rows: List[Dict], path: str = "data/history.csv"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(p, mode="a", header=not p.exists(), index=False)


def load(path: str = "data/history.csv") -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(p, parse_dates=["timestamp"])


def predict(key: str, path: str = "data/history.csv", threshold: float = 15.0) -> Optional[Dict]:
    """Estimate days until delta-T reaches the 'major' threshold. Needs >= 3 inspections."""
    df = load(path)
    df = df[df["equipment_key"] == key].sort_values("timestamp")
    if len(df) < 3:
        return {"status": f"Need at least 3 inspections for a trend (have {len(df)})."}
    days = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds().to_numpy() / 86400
    if days[-1] - days[0] < 0.5:
        return {"status": "Inspections too close in time for a trend."}
    slope, intercept = np.polyfit(days, df["delta_used"].to_numpy(), 1)
    now = slope * days[-1] + intercept
    if now >= threshold:
        return {"status": "Already at or above the major threshold.", "slope_per_day": float(slope)}
    if slope <= 0.01:
        return {"status": "Stable or improving.", "slope_per_day": float(slope)}
    return {
        "status": f"Rising {slope:.2f} C/day - estimated {(threshold - now) / slope:.0f} days to reach {threshold:.0f} C rise.",
        "slope_per_day": float(slope),
        "days_to_major": float((threshold - now) / slope),
    }
