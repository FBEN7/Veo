"""Descriptive events on the M7 / M9 test windows (after protocol v5 part 2
was scored): sweep_events_v2.apply's frozen configuration (events_v3.json),
the same code path, with two changes only -- it runs on a test match
(opened through src.unseal, logged), and the ball rows' threshold is their
own detector's calibrated tau (v2: 0.25, as frozen; v3: 0.10), since
events_v3 froze v2's.

    python apply.py WINDOW NAME ROWS.parquet CKPT.calib.json
"""
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/Veo")
import pandas as pd  # noqa: E402

import score_soccernet as sc  # noqa: E402
import sweep_events_v2 as se  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.unseal import unsealed  # noqa: E402

REC = Path("/home/user/Veo/research/session_scratchpad/ball/events_v3.json")
w, name, rows_path, calib_path = sys.argv[1:5]
digest = Path(f"{REC}.sha256").read_text().split()[0]
assert se.sha256_of(REC) == digest == "fd2600613cf3601296936de14ae884b03b4cb5d7cbd6580749e00eb8e77aa931"
record = json.loads(REC.read_text())
m = se.match_of(w)
if m in {"128057", "132831"} and not unsealed(m, f"events descriptive {name}"):
    raise SystemExit(f"{w}: test match (VEO_UNSEAL needed)")
calib = json.loads(Path(calib_path).read_text()) | {"path": calib_path}
tau = calib["tau"]
meta = se.balls_meta(w, rows_path, calib, tau)
src = Path(f"/home/user/Veo/output_{w}") / se.SRC
dst = Path(f"/home/user/Veo/output_{w}") / name
cal = st.calibration(m)
with contextlib.redirect_stdout(io.StringIO()):
    ref = sc.merged_from_cached(src, cal).attrs["player_hash"]
rows = pd.read_parquet(rows_path)
sc.swap_ball_rows(src, dst, rows)
frozen = {**se.BASE, **record["frozen"]}
select, top_k = frozen["select"], frozen["top_k"]
link_params = {k: frozen[k] for k in se.LINK_DEFAULT} if select == "link" else None
rules = {k: v for k, v in se.changed(frozen).items() if k in se.V1_RULES}
with contextlib.redirect_stdout(io.StringIO()):
    events, merged = sc.events_from_cached(dst, cal, select=select, link_params=link_params,
                                           top_k=top_k, tau=tau, event_params=rules)
se.check_player_hash(w, merged.attrs["player_hash"], ref, record)
sc.write_events(dst / "events.json", events)
merged.to_parquet(dst / "tracks_merged.parquet")
(dst / "applied.json").write_text(json.dumps({
    "events_v3": str(REC), "sha256": digest, "window": w, "deviation": "tau of the rows' own calibration",
    "balls": {"path": rows_path, "sha256": se.sha256_of(rows_path), **meta},
    "select": select, "link_params": link_params, "top_k": top_k, "tau": tau, "rules": rules,
    "player_hash": merged.attrs["player_hash"], "source_player_hash": ref}, indent=1))
print(f"{dst}: {len(events)} events {pd.Series([e['event_type'] for e in events]).value_counts().to_dict()}")
