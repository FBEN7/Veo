import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import ball_track as bt, soccertrack_v2 as st
from src.ball_heatmap import pitch_to_image
for w in ["st2_118578_1st_f030000", "st2_118578_2nd_f015000"]:
    path = f"/home/user/Veo/.cache/v2_balls_{w}.parquet"
    side = json.load(open(path + ".meta.json"))
    done = np.array(side["frames"], int); df = pd.read_parquet(path)
    cal = st.calibration(side["match"])
    picks = bt.link_picks(df, cal, 0.25, frames=done)
    f = picks.frame.to_numpy(); d = np.diff(f)
    sp, em, ep, n_fr = [], [], [], 0
    for i in np.flatnonzero((d >= 2) & (d <= 6)):
        A, B = picks.iloc[i], picks.iloc[i + 1]
        ga = st.image_to_pitch([[A.u, A.v + A["size"] / 2]], cal)[0]
        gb = st.image_to_pitch([[B.u, B.v + B["size"] / 2]], cal)[0]
        sp.append(np.hypot(*(gb - ga)) / (d[i] / 25))
        for k in range(1, int(d[i])):
            w_ = k / d[i]
            img = np.array([A.u + w_ * (B.u - A.u), A.v + A["size"]/2 + w_ * (B.v + B["size"]/2 - A.v - A["size"]/2)])
            gi = st.image_to_pitch([img], cal)[0]
            em.append(np.hypot(*(gi - (ga + w_ * (gb - ga)))))
            ep.append(np.hypot(*(img - pitch_to_image(ga + w_ * (gb - ga), cal)[0])))
    sp, em, ep = map(np.array, (sp, em, ep))
    print(w, f"G<=6 pairs {len(sp)}: speed>35 m/s {int((sp>35).sum())}, >25 {int((sp>25).sum())}; "
          f"fill frames {len(ep)}: >6 px {int((ep>6).sum())}, >12 px {int((ep>12).sum())}; "
          f"metres: p99 {np.percentile(em,99):.2f} max {em.max():.2f}, >0.1 m {int((em>0.1).sum())}")
