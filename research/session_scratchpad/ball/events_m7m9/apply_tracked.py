"""Events with the tracked v3 ball (step 1 of the events plan): v3's dense rows
with the spot marks dropped (.cache/v3m_balls_<w>.parquet), tracked at stride 1
with protocol v4-r1's settings and v3's tau 0.10 (`ball_track.track`, as v3P in
protocol v5 part 2), written as ball rows (.cache/v3p_balls_<w>.parquet), then
the frozen events_v3 rules on them with select="given" (no second selection).

    python apply_tracked.py WINDOW NAME [--rules frozen|v1]
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/Veo")
import pandas as pd  # noqa: E402

import score_soccernet as sc  # noqa: E402
import sweep_events_v2 as se  # noqa: E402
from detect_ball_heatmap import read_sidecar  # noqa: E402
from src import ball_track as bt  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.unseal import unsealed  # noqa: E402

V4R1 = {"tau": 0.10, "top_k": 4, "c_null": 1.0, "c_static": 2.0, "gap": 6,
        "max_fill_speed": 35.0, "fallback": True}
REC = Path("/home/user/Veo/research/session_scratchpad/ball/events_v3.json")


def tracked_rows(w: str, cal: dict, fallback: bool = True) -> Path:
    tag = "" if fallback else "_nofb"
    out = Path(f"/home/user/Veo/.cache/v3p{tag}_balls_{w}.parquet")
    if out.exists():
        return out
    src = Path(f"/home/user/Veo/.cache/v3m_balls_{w}.parquet")
    meta = read_sidecar(src)
    rows = pd.read_parquet(src)
    info = {}
    tr = bt.track(rows, cal, frames=meta["frames"], info=info, **{**V4R1, "fallback": fallback})
    balls = pd.DataFrame({"frame": tr.frame.astype(int), "time_s": tr.frame / 25.0, "track_id": -1,
                          "cls": "ball", "px": tr.u, "py": tr.v, "crop_h": tr["size"], "crop_w": tr["size"],
                          "detection_method": "v3_tracked_" + tr.source.astype(str), "confidence": tr.p})
    balls.to_parquet(out)
    Path(f"{out}.json").write_text(json.dumps({"from": str(src), "parquet sha256 of source": se.sha256_of(src),
                                               "params": {**V4R1, "fallback": fallback}, "counts": info}, indent=1))
    print(f"{w}: tracked ball on {len(balls)} of {len(meta['frames'])} frames {info}")
    return out


ap = argparse.ArgumentParser()
ap.add_argument("window")
ap.add_argument("name")
ap.add_argument("--rules", choices=("frozen", "v1"), default="frozen")
ap.add_argument("--no-fallback", action="store_true", help="no raw top candidate where the tracker has no pick")
ap.add_argument("--event-params", help="JSON of event rule overrides (on top of --rules)")
args = ap.parse_args()
w = args.window
m = se.match_of(w)
if m in {"128057", "132831"} and not unsealed(m, f"events tracked {args.name}"):
    raise SystemExit(f"{w}: test match (VEO_UNSEAL needed)")
record = json.loads(REC.read_text())
assert se.sha256_of(REC) == "fd2600613cf3601296936de14ae884b03b4cb5d7cbd6580749e00eb8e77aa931"
cal = st.calibration(m)
rows_path = tracked_rows(w, cal, fallback=not args.no_fallback)
src = Path(f"/home/user/Veo/output_{w}") / se.SRC
dst = Path(f"/home/user/Veo/output_{w}") / args.name
with contextlib.redirect_stdout(io.StringIO()):
    ref = sc.merged_from_cached(src, cal).attrs["player_hash"]
sc.swap_ball_rows(src, dst, pd.read_parquet(rows_path))
cfg = {**se.BASE, **record["frozen"]} if args.rules == "frozen" else dict(se.BASE)
rules = {k: v for k, v in se.changed(cfg).items() if k in se.V1_RULES}
rules.update(json.loads(args.event_params or "{}"))
with contextlib.redirect_stdout(io.StringIO()):
    events, merged = sc.events_from_cached(dst, cal, select="given", event_params=rules)
assert merged.attrs["player_hash"] == ref, "players changed"
sc.write_events(dst / "events.json", events)
merged.to_parquet(dst / "tracks_merged.parquet")
(dst / "applied.json").write_text(json.dumps({
    "window": w, "balls": {"path": str(rows_path), "sha256": se.sha256_of(rows_path)}, "select": "given",
    "tracker": {**V4R1, "fallback": not args.no_fallback}, "rules": rules, "player_hash": ref}, indent=1))
print(f"{dst}: {len(events)} events {pd.Series([e['event_type'] for e in events]).value_counts().to_dict()}")
