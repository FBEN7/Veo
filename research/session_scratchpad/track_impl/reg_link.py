"""Regression: ball_link.link (new) == HEAD's link with players given, on the
cached U-E pipeline runs; counts and equality only."""
import importlib.util
import json
import sys

import pandas as pd

sys.path.insert(0, "/home/user/Veo")
from src import ball_link as new  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "src.ball_link_head",
    "/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/track_impl/head/ball_link_head.py")
old = importlib.util.module_from_spec(spec)
old.__package__ = "src"
spec.loader.exec_module(old)

for w in ("st2_118578_1st_f030000", "st2_118578_2nd_f015000"):
    t = pd.read_parquet(f"/home/user/Veo/output_{w}/events_v2_sweep/tracks_teams.parquet")
    cal = st.calibration("118578")
    balls = t[t.cls == "ball"]
    balls = balls[balls.confidence >= 0.25]
    balls = (balls.sort_values("confidence", ascending=False, kind="stable")
             .groupby("frame", sort=False).head(4))
    players = t[t.cls == "player"]
    for cn, cs in ((1.0, 2.0), (3.0, 6.0), (1.0, 0.0)):
        a = old.link(balls, cal, players, c_null=cn, c_static=cs, fps=25, tau=0.25)
        b = new.link(balls, cal, players, c_null=cn, c_static=cs, fps=25, tau=0.25)
        print(w, f"c_null {cn} c_static {cs}: {len(balls)} candidates, {len(players)} "
              f"player rows -> head {len(a)}, new {len(b)} kept; identical: {a.equals(b)}")
        assert a.equals(b)
    # Dense rows too (another input), with the players.
    d = pd.read_parquet(f"/home/user/Veo/.cache/v2_balls_{w}.parquet")
    d = d[d.confidence >= 0.25]
    d = d.sort_values("confidence", ascending=False, kind="stable").groupby("frame", sort=False).head(4)
    a = old.link(d, cal, players, tau=0.25)
    b = new.link(d, cal, players, tau=0.25)
    print(w, f"dense rows: {len(d)} -> head {len(a)}, new {len(b)}; identical: {a.equals(b)}")
    assert a.equals(b)
    # players=None == an empty player frame in HEAD's link
    a = old.link(d, cal, players.iloc[:0], tau=0.25)
    b = new.link(d, cal, None, tau=0.25)
    print(w, f"players=None == HEAD with no player rows: {a.equals(b)} ({len(b)} kept)")
    assert a.equals(b)
