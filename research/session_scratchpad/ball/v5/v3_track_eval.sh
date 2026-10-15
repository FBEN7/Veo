#!/bin/sh
# Step 3: tracked v3 (stride 1 and 2) and tracked v2 (stride 2) on M6's second half,
# scored against the verified ball (registered: v3_track_selection.json).
cd /home/user/Veo
V=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball
W1=st2_118578_2nd_f015000; W2=st2_118578_2nd_f037500
python - <<'PY' || exit 1
import pandas as pd
from detect_ball_heatmap import read_sidecar, write_sidecar
for w in ("st2_118578_2nd_f015000", "st2_118578_2nd_f037500"):
    src = f".cache/v3_balls_{w}.parquet"
    meta = read_sidecar(src)
    assert meta["stride"] == 1 and not meta["extra_frames"]
    df = pd.read_parquet(src)
    out = f".cache/v3s2_balls_{w}.parquet"
    df[df.frame % 2 == 0].to_parquet(out)
    frames = [f for f in meta["frames"] if f % 2 == 0]
    keep = {k: v for k, v in meta.items() if k not in ("frames", "parquet sha256", "stride", "requested", "chunks")}
    write_sidecar(out, frames, **keep, stride=2, requested=len(frames),
                  derived_from={"parquet": src, "parquet sha256": meta["parquet sha256"], "rule": "even frames"})
    print(w, len(df), "rows ->", out, len(frames), "frames")
PY
P3='{"tau": 0.1, "top_k": 4, "c_null": 1.0, "c_static": 2.0, "gap": 6, "max_fill_speed": 35.0, "fallback": true}'
python eval_ball_detector.py --reuse $S/U_a.json $S/U_b.json $V/v3eval_b_$W1.json $V/v3eval_b_$W2.json \
  --windows $W1 $W2 \
  --track 'v3T1=.cache/v3_balls_{window}.parquet' 'v3T2=.cache/v3s2_balls_{window}.parquet' 'v2T2=.cache/v2_balls_{window}.parquet' \
  --track-params "v3T1=$P3" "v3T2=$P3" "v2T2=$S/track_params.json" \
  --save $V/v3track.json > $V/v3track.log 2>&1 || exit 1
python score_verified_top1.py --scores $S/U_a.json $S/U_b.json $V/v3eval_b_$W1.json $V/v3eval_b_$W2.json $V/v3track.json \
  --windows $W1 $W2 --detectors v3T1 v3T2 ball_v3b.1500.pt v2T2 ball_v2.4000.pt \
  --reference ball_v3b.1500.pt --out .cache/soccertrack_v2/ball_verified/v3_track_scores.json > $V/v3track_score.log 2>&1 || exit 1
echo track-eval-done
