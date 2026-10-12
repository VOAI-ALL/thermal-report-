import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from thermal_ai.detector import Detection
from thermal_ai.diagnosis import analyze


def scene(hot_delta):
    t = np.full((200, 300), 30.0, dtype=np.float32)
    for cx, d in [(50, 5), (150, hot_delta), (250, 5)]:
        t[80:120, cx - 20:cx + 20] += 3
        t[85:90, cx - 3:cx + 3] += d      # small hotspot near the top edge (terminal)
    dets = [Detection("terminal_block", (cx - 25, 60, cx + 25, 140)) for cx in (50, 150, 250)]
    return t, dets


def test_loose_connection_detected():
    t, dets = scene(30)
    f = analyze(dets, t, ambient=30)
    worst = max(f, key=lambda x: x.severity)
    assert worst.bbox[0] == 125 and worst.severity == 3
    assert "terminal" in worst.fault.lower()
    assert all(x.severity < 3 for x in f if x is not worst)


def test_normal_scene():
    t, dets = scene(1)
    f = analyze(dets, t, ambient=30)
    assert max(x.severity for x in f) <= 2


def test_all_phases_hot_is_overload():
    t = np.full((200, 300), 30.0, dtype=np.float32)
    dets = []
    for cx in (50, 150, 250):
        t[70:130, cx - 25:cx + 25] += 25   # whole area hot on all phases
        dets.append(Detection("cable", (cx - 25, 70, cx + 25, 130)))
    f = analyze(dets, t, ambient=30)
    assert all(x.severity <= 2 for x in f)


def test_load_correction_raises_severity():
    t, dets = scene(8)
    a = max(x.severity for x in analyze(dets, t, 30))
    b = max(x.severity for x in analyze(dets, t, 30, load_pct=50))
    assert b >= a


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
