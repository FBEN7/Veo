"""Read-only: completed ground passes (eval_ball_detector criteria) per half
and per minute, for halves whose game state is cached."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import soccertrack_v2 as st
t0 = time.time()
rows = []
for m, h in (("117093", 1), ("132877", 1), ("118577", 1), ("118575", 1), ("118576", 1), ("118576", 2), ("118578", 1), ("118578", 2), ("128058", 1), ("128058", 2)):
    ev = st.events(m); ev = ev[ev.half == h].sort_values("frame").reset_index(drop=True)
    g = pd.read_parquet(f"/home/user/Veo/.cache/soccertrack_v2/gsr/{m}_{st._half(h)}.parquet", columns=["frame", "player_id", "x", "y"])
    g = g[g.player_id.notna()].copy(); g["player_id"] = g.player_id.astype(str)
    e = ev.merge(g, on=["frame", "player_id"], how="left")
    E = e.to_dict("records")
    for a, b in zip(E[:-1], E[1:]):
        if a["label"] != "PASS" or b["side"] != a["side"] or b["player_id"] == a["player_id"] or pd.isna(a["x"]) or pd.isna(b["x"]):
            continue
        L = np.hypot(b["x"] - a["x"], b["y"] - a["y"]); d = (b["frame"] - a["frame"]) / 25
        if L >= 10 and d >= 0.8:
            rows.append(dict(m=m, h=h, minute=int(a["frame"] // 1500), L=L, d=d))
    nev = len(ev); npc = int(ev.label.isin(["PASS", "HIGH PASS", "CROSS", "DRIVE"]).sum())
    rows.append(dict(m=m, h=h, minute=-1, L=nev, d=npc))
df = pd.DataFrame(rows)
ff = df[df.minute >= 0]
tot = ff.groupby(["m", "h"]).size()
mins = ff.groupby(["m", "h"]).minute.max() + 1
print("completed ground passes per half:\n", pd.concat([tot.rename("passes"), mins.rename("minutes"), (tot / mins).round(2).rename("per_min"),
      df[df.minute < 0].set_index(["m", "h"]).L.rename("events"), df[df.minute < 0].set_index(["m", "h"]).d.rename("pass+carry labels")], axis=1))
print("M2/M10 1st half: passes in minutes 10-12 (used V):", ff[(ff.m.isin(["117093", "132877"])) & (ff.minute.between(10, 12))].groupby("m").size().to_dict())
ev2 = {m: st.events(m).groupby("half").size().to_dict() for m in ("117093", "132877", "118577")}
print("events per half (for 2nd-half estimates):", ev2)
print(f"{time.time()-t0:.0f}s")
