"""GT-free: the filled pairs with the largest picture- vs pitch-linear gap:
pitch distance and implied speed between the two picks, px span, scale."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from src import ball_track as bt, soccertrack_v2 as st
from src.ball_heatmap import pitch_to_image, local_scale
for w in ["st2_118578_1st_f030000", "st2_118578_2nd_f015000"]:
    path = f"/home/user/Veo/.cache/v2_balls_{w}.parquet"
    side = json.load(open(path + ".meta.json"))
    done = np.array(side["frames"], int); df = pd.read_parquet(path)
    cal = st.calibration(side["match"])
    picks = bt.link_picks(df, cal, 0.25, frames=done)
    f = picks.frame.to_numpy(); d = np.diff(f)
    out = []
    for i in np.flatnonzero((d >= 2) & (d <= 12)):
        A, B = picks.iloc[i], picks.iloc[i + 1]
        ga = st.image_to_pitch([[A.u, A.v + A["size"] / 2]], cal)[0]
        gb = st.image_to_pitch([[B.u, B.v + B["size"] / 2]], cal)[0]
        e = 0
        for k in range(1, int(d[i])):
            w_ = k / d[i]
            img = np.array([A.u + w_ * (B.u - A.u), A.v + A["size"]/2 + w_ * (B.v + B["size"]/2 - A.v - A["size"]/2)])
            e = max(e, float(np.hypot(*(img - pitch_to_image(ga + w_ * (gb - ga), cal)[0]))))
        dm = float(np.hypot(*(gb - ga)))
        out.append((e, int(d[i]), dm, dm / (d[i] / 25), float(np.hypot(B.u - A.u, B.v - A.v)), ga.round(1).tolist(), gb.round(1).tolist()))
    out.sort(reverse=True)
    print(w)
    for e, dd, dm, v, pxs, ga, gb in out[:8]:
        print(f"  fill err {e:6.1f} px  gap {dd:2d}  pitch {dm:5.1f} m  speed {v:5.1f} m/s  px span {pxs:6.1f}  {ga} -> {gb}")
    sp = np.array([o[3] for o in out]); dist = np.array([o[2] for o in out]); gg = np.array([o[1] for o in out])
    print("  filled pairs", len(out), "| speed > 35 m/s:", int((sp > 35).sum()), "| > 60:", int((sp > 60).sum()),
          "| pitch jump >= 5 m at gap<=6:", int(((dist >= 5) & (gg <= 6)).sum()), "| at gap<=12:", int((dist >= 5).sum()))
