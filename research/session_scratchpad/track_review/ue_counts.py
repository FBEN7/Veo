"""GT-free implementation checks of src/ball_track on the two U-E dense
parquets: counts and geometry only (no labels, no measure)."""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/Veo")
from src import ball_track as bt  # noqa: E402
from src import ball_link as bl  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.ball_heatmap import pitch_to_image  # noqa: E402

W = ["st2_118578_1st_f030000", "st2_118578_2nd_f015000"]
for w in W:
    path = f"/home/user/Veo/.cache/v2_balls_{w}.parquet"
    side = json.load(open(path + ".meta.json"))
    done = np.array(side["frames"], int)
    df = pd.read_parquet(path)
    cal = st.calibration(side["match"])
    tau = side["calib"]["tau"]
    print(w, "rows", len(df), "processed", len(done), "extra", len(side["extra_frames"]),
          "stride", side["stride"], "odd processed", int((done % 2).sum()))
    picks = bt.link_picks(df, cal, tau, top_k=4, c_null=1, c_static=2, frames=done)
    picks_none = bt.link_picks(df, cal, tau, top_k=4, c_null=1, c_static=2)
    # players None vs an explicit empty players table (revert option)
    cand = bt.candidates(df, tau, 4)
    k_none = bl.link(cand, cal, None, c_null=1, c_static=2, tau=tau, frames=done)
    k_empty = bl.link(cand, cal, cand.iloc[:0], c_null=1, c_static=2, tau=tau, frames=done)
    print("  picks", len(picks), "| frames=None picks", len(picks_none),
          "same set:", set(picks.frame) == set(picks_none.frame),
          "| players None == empty table:", k_none.equals(k_empty))
    f = picks.frame.to_numpy()
    d = np.diff(f)
    print("  pick spacing counts (frames):", dict(zip(*np.unique(np.minimum(d, 20), return_counts=True))))
    for G in (6, 12):
        tr = bt.fill_gaps(picks, G)
        interp = tr[tr.source == "interpolated"]
        odd = tr[tr.frame % 2 == 1]
        print(f"  G={G}: rows {len(tr)}, interpolated {len(interp)}, "
              f"odd rows all interpolated: {(odd.source == 'interpolated').all()}, "
              f"processed frames filled (linker null there): {int((interp.frame % 2 == 0).sum())}")
    # picture-linear vs pitch-linear interpolation, at the filled frames (G = 12)
    a_idx = np.flatnonzero((d >= 2) & (d <= 12))
    errs, errs6 = [], []
    for i in a_idx:
        A, B = picks.iloc[i], picks.iloc[i + 1]
        ga = st.image_to_pitch([[A.u, A.v + A["size"] / 2]], cal)[0]
        gb = st.image_to_pitch([[B.u, B.v + B["size"] / 2]], cal)[0]
        for k in range(1, int(d[i])):
            w_ = k / d[i]
            img = np.array([A.u + w_ * (B.u - A.u),
                            A.v + A["size"] / 2 + w_ * (B.v + B["size"] / 2 - A.v - A["size"] / 2)])
            pit = pitch_to_image(ga + w_ * (gb - ga), cal)[0]
            e = float(np.hypot(*(img - pit)))
            errs.append(e)
            if d[i] <= 6:
                errs6.append(e)
    errs, errs6 = np.array(errs), np.array(errs6)
    print(f"  picture- vs pitch-linear fill, px: G<=6 n={len(errs6)} median {np.median(errs6):.2f} "
          f"p99 {np.percentile(errs6, 99):.2f} max {errs6.max():.2f}; G<=12 n={len(errs)} "
          f"max {errs.max():.2f}")
    # speed between consecutive picks <= 12 frames apart (pitch)
    g = bl.ground_points(df.loc[[]].assign(), cal) if False else None
