import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/user/Veo")
from eval_soccertrack_v2 import boxes, match_feet, truth, FOOT_PX
from src.paths import DATA_DIR

for window in sys.argv[1:]:
    root = DATA_DIR / "soccertrack_v2"
    gt = truth(root / f"{window}_gt.parquet")
    frames = sorted(f for f in gt.frame.unique() if f % 10 == 0)
    gt = gt[gt.frame.isin(frames)]
    t = pd.read_parquet(f"/home/user/Veo/output_{window}/tracks.parquet")
    det = boxes(t[(t.cls == "player") & t.frame.isin(frames)].copy())
    alld = det.copy()
    det["person"] = match_feet(det, gt)
    unmatched = det[det.person.isna()]
    det = det.dropna(subset=["person"])
    feet = {f: g for f, g in gt.groupby("frame")}
    um = {f: g for f, g in unmatched.groupby("frame")}
    keep, rival = [], []
    for r in det.itertuples():
        g = feet[r.frame]
        d = np.hypot((g.x0 + g.x1) / 2 - (r.x0 + r.x1) / 2, g.y1 - r.y1).to_numpy()
        own = float(d[(g.person == r.person).to_numpy()][0])
        others = d[(g.person != r.person).to_numpy()]
        radius = min(FOOT_PX, 0.6 * r.crop_h)
        k = own <= radius and bool((others >= 2 * radius).all())
        keep.append(k)
        # an unmatched detection (referee / duplicate / FP) whose feet are within
        # 2r of the labelled detection's feet: two candidates for one name
        u = um.get(r.frame)
        if u is None or not k:
            rival.append(False); continue
        gp = g[g.person == r.person].iloc[0]
        gx, gy = (gp.x0 + gp.x1) / 2, gp.y1
        du = np.hypot((u.x0 + u.x1) / 2 - gx, u.y1 - gy).to_numpy()
        rival.append(bool((du <= radius).any()))
    keep = np.array(keep); rival = np.array(rival)
    print(window, "matched", len(det), "kept", keep.sum(), "kept with an unmatched det within r of the GT feet", rival.sum(), f"({rival.sum()/max(keep.sum(),1):.1%})")
