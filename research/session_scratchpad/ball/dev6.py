import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
tot = {}
for m, h in (("118575", 1), ("118576", 1), ("118577", 1), ("118578", 1), ("128058", 1), ("118576", 2), ("118578", 2), ("128058", 2), ("117093", 1), ("132877", 1)):
    ev = st.events(m); ev = ev[ev.half == h].sort_values("frame").reset_index(drop=True)
    g = pd.read_parquet(f"/home/user/Veo/.cache/soccertrack_v2/gsr/{m}_{st._half(h)}.parquet", columns=["frame", "player_id", "x", "y"])
    g = g[g.player_id.notna()]; g["player_id"] = g.player_id.astype(str)
    e = ev.merge(g, on=["frame", "player_id"], how="left")
    lab = e.label.value_counts().to_dict()
    n = {"ground": 0, "ground_frames": 0, "high": 0, "cross": 0, "completed_high": 0, "short": 0}
    durs, Ls = [], []
    for i in range(len(e) - 1):
        a, b = e.iloc[i], e.iloc[i + 1]
        if a.label not in ("PASS", "HIGH PASS", "CROSS"):
            continue
        comp = b.side == a.side and b.player_id != a.player_id and pd.notna(a.x) and pd.notna(b.x)
        if not comp:
            continue
        L = np.hypot(b.x - a.x, b.y - a.y); d = (b.frame - a.frame) / 25
        if a.label == "PASS":
            if L >= 10 and d >= 0.8:
                n["ground"] += 1; durs.append(d); Ls.append(L); n["ground_frames"] += int(0.7 * (b.frame - a.frame))
            elif L >= 5 and d >= 0.5:
                n["short"] += 1
        else:
            n["completed_high"] += 1
    n["HIGH PASS"] = lab.get("HIGH PASS", 0); n["CROSS"] = lab.get("CROSS", 0); n["PASS"] = lab.get("PASS", 0); n["DRIVE"] = lab.get("DRIVE", 0)
    n["events"] = len(e); n["dur_med"] = round(float(np.median(durs)), 2); n["L_med"] = round(float(np.median(Ls)), 1)
    n["missing_xy"] = int(e.x.isna().sum())
    print(m, h, n)
