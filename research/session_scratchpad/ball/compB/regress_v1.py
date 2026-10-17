"""v1 detect() before/after the refactor, on a few frames of training windows."""
import importlib.util, json, sys, time
import cv2, numpy as np, torch
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
from src import ball_heatmap as new
from src.paths import DATA_DIR

spec = importlib.util.spec_from_file_location(
    "src.ball_heatmap_orig", "/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/compB/ball_heatmap_v1_orig.py")
old = importlib.util.module_from_spec(spec); old.__package__ = "src"; spec.loader.exec_module(old)
torch.set_num_threads(4)
sd = torch.load("/home/user/Veo/.cache/ball_heatmap.3000.pt")
m_old, m_new = old.build(), new.build()
m_old.load_state_dict(sd); m_new.load_state_dict(sd)
n = 0
for w, frames in (("st2_118575_1st_f015000", (100, 2000)), ("st2_118577_1st_f030000", (300,)),
                  ("st2_118576_1st_f030000", (1500,))):
    info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
    cal = st.calibration(info["match"])
    cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{w}.mp4"))
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read(); assert ok
        for kw in (dict(tau=0.01, top_k=8, cal=cal), dict(tau=0.7, top_k=4), dict(tau=0.99999)):
            a = old.detect(m_old, im, info["pitch"], **kw)
            b = new.detect(m_new, im, info["pitch"], **kw)
            assert a.shape == b.shape and np.array_equal(a, b, equal_nan=True), (w, f, kw)
            n += 1
print(f"v1 detect identical before/after on {n} (frame, setting) pairs")
