import sys, io, contextlib, json, shutil, time
sys.path.insert(0, "/home/user/Veo")
from pathlib import Path
import pandas as pd, numpy as np
import score_soccernet as sc
from detect_ball_heatmap import V2_COLUMNS
from src import soccertrack_v2 as st
w = "st2_118575_1st_f015000"
info = json.loads(open(f"/home/user/Veo/data/soccertrack_v2/{w}.json").read())
cal = st.calibration(info["match"])
src = Path(f"/home/user/Veo/output_{w}/events_v1")
scr = Path("/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/review_F")
dst = scr / "swap_dst"
raw = pd.read_parquet(src / "tracks.parquet")
balls = raw[raw.cls == "ball"].copy()
# fake v2 rows: same positions, confidence as p_cal, size, logit
rng = np.random.default_rng(0)
balls["confidence"] = rng.uniform(0.05, 0.99, len(balls))
balls["crop_h"] = 6.0
balls["crop_w"] = np.nan
balls["detection_method"] = "heatmap_v2"
balls["logit"] = np.log(balls.confidence / (1 - balls.confidence))
balls["p_raw"] = balls.confidence
balls["n_bg"] = 15
# add an extra odd frame row (should be dropped)
extra = balls.iloc[:1].copy(); extra["frame"] = 1
v2 = pd.concat([balls, extra])[list(V2_COLUMNS)].astype(V2_COLUMNS).reset_index(drop=True)
v2.to_parquet(scr / "fake_v2.parquet")
v2 = pd.read_parquet(scr / "fake_v2.parquet")
print(v2.dtypes.to_dict())
sc.swap_ball_rows(src, dst, v2)
# avoid decode: grass file = teamed with NaN grass
teamed = pd.read_parquet(dst / "tracks_teams.parquet")
g = teamed.copy(); g["grass_frac"] = np.nan
g.to_parquet(dst / "tracks_grass.parquet")
with contextlib.redirect_stdout(io.StringIO()):
    h_src = sc.merged_from_cached(src, cal).attrs["player_hash"]
t0 = time.time()
for sel, k in (("v1", 4), ("link", 4), ("link", 1), ("v1", 1)):
    with contextlib.redirect_stdout(io.StringIO()):
        m = sc.merged_from_cached(dst, cal, select=sel, top_k=k, tau=0.3,
                                  link_params={"c_null": 1.0, "c_static": 2.0} if sel == "link" else None)
    b = m[m.cls == "ball"]
    print(sel, k, m.attrs["player_hash"] == h_src, b.frame.is_unique, len(b), (b.frame % 2).sum(), round(time.time() - t0, 1))
