"""Central settings. Edit these to match your site standards."""

CLASSES = ["CT", "VT", "busbar", "switchgear", "MCB", "MCCB", "cable", "terminal_block"]

# Absolute temperature limits in deg C: (caution, critical).
# DEFAULTS ONLY - replace with the manufacturer's rated limits for your equipment.
ABS_LIMITS = {
    "CT": (80, 100),
    "VT": (80, 100),
    "busbar": (90, 105),
    "switchgear": (80, 100),
    "MCB": (80, 100),
    "MCCB": (80, 100),
    "cable": (70, 90),
    "terminal_block": (90, 105),
    "unknown": (90, 105),
}

# Hotspot "shape" thresholds (fraction of the equipment box that is hot)
LOCALIZED_FRACTION = 0.12   # below this -> small, point-like hotspot (connection problem)
WIDESPREAD_FRACTION = 0.35  # above this -> large area hot (overload / imbalance)

# Load correction only trusted at or above this load (NETA recommends >= 40 %)
MIN_LOAD_PCT = 40
MAX_LOAD_FACTOR = 4.0

SEVERITY_NAMES = {
    0: "Normal",
    1: "Possible deficiency - monitor",
    2: "Probable deficiency - repair soon",
    3: "MAJOR - repair immediately",
}
