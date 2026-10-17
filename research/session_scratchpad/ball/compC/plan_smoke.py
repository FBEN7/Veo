import sys, json
sys.path.insert(0, "/home/user/Veo")
import numpy as np, pandas as pd
import build_ball_strips as bs
from src.paths import DATA_DIR
w = "st2_118577_1st_f030000"
info = json.loads((DATA_DIR / "soccertrack_v2" / f"{w}.json").read_text())
h = bs.Half("118577", 1, window=w)
poly = np.asarray(info["pitch"], np.float32)
box = bs.pitch_box((info["height"], info["width"]), poly)
print("box", box)
t = pd.DataFrame(bs.samples(h, bs.KINDS, box, poly))
t = t[t.frame.between(30000, 34499)]
print(t.groupby("kind").size().to_dict())
for lo, hi in ((30000, 30500), (30000, 31200), (30400, 31400)):
    s = t[t.frame.between(lo + 2, hi - 2)]
    print(lo, hi, s.groupby("kind").size().to_dict())
print(t[t.kind.isin(["flight", "lofted", "dribble"])][["frame", "kind", "category", "L", "s"]].head(20).to_string())
