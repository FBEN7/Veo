import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
for m, h in (("118575", 1), ("128058", 1), ("118578", 2)):
    ev = st.events(m); ev = ev[ev.half == h].sort_values("frame").reset_index(drop=True)
    g = pd.read_parquet(f"/home/user/Veo/.cache/soccertrack_v2/gsr/{m}_{st._half(h)}.parquet", columns=["frame", "player_id", "x", "y"])
    g = g[g.player_id.notna()].copy(); g["player_id"] = g.player_id.astype(str)
    E = ev.merge(g, on=["frame", "player_id"], how="left").to_dict("records")
    c = dict(ground=0, intercepted=0, long35=0, lofted=0)
    for a, b in zip(E[:-1], E[1:]):
        if pd.isna(a["x"]) or pd.isna(b["x"]):
            continue
        L = np.hypot(b["x"] - a["x"], b["y"] - a["y"]); d = (b["frame"] - a["frame"]) / 25
        if a["label"] == "PASS" and L >= 10 and 0.8 <= d:
            if b["side"] == a["side"] and b["player_id"] != a["player_id"]:
                c["ground"] += 1; c["long35"] += int(L >= 35)
            elif b["side"] != a["side"] and b["label"] in ("PASS", "DRIVE", "HIGH PASS") and d <= 4:
                c["intercepted"] += 1
        if a["label"] in ("HIGH PASS", "CROSS") and d >= 0.8:
            c["lofted"] += 1
    print(m, h, c)
