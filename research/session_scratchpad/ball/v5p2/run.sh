#!/bin/sh
# Protocol v5 part 2, CPU part: cut the two test windows, run the detectors, build the
# blind tiles. Needs VEO_UNSEAL = sha256 of v5p2_registration.json. Restart-proof:
# each step is skipped when its output exists. Prints counts only.
set -u
cd /home/user/Veo
R=research/session_scratchpad/ball/v5p2
O=.cache/soccertrack_v2/ball_verified/v5p2
mkdir -p $O
W1=st2_128057_1st_f030000; W2=st2_132831_2nd_f030000
[ -f data/soccertrack_v2/$W1.json ] || python fetch_soccertrack_v2.py --match 128057 --half 1 --start 20 --minutes 3 > $O/fetch_$W1.log 2>&1 || exit 1
[ -f data/soccertrack_v2/$W2.json ] || python fetch_soccertrack_v2.py --match 132831 --half 2 --start 20 --minutes 3 > $O/fetch_$W2.log 2>&1 || exit 1
echo windows-cut
for W in $W1 $W2; do
  [ -f $O/eval_$W.json ] || { python eval_ball_detector.py --windows $W \
      --v2 .cache/ball_v2.4000.pt .cache/ball_v3b.1500.pt --heatmap .cache/ball_heatmap.3000.pt \
      --sealed $O/eval_$W.json.part > $O/eval_$W.log 2>&1 && mv $O/eval_$W.json.part $O/eval_$W.json; } || exit 1
  [ -f .cache/v3_balls_$W.parquet.meta.json ] || python $R/dense_v5p2.py $W > $O/dense_$W.log 2>&1 || exit 1
  [ -f .cache/v3m_balls_$W.parquet.meta.json ] || python $R/marks.py $W > $O/marks_$W.log 2>&1 || exit 1
done
echo detectors-done
P3='{"tau": 0.1, "top_k": 4, "c_null": 1.0, "c_static": 2.0, "gap": 6, "max_fill_speed": 35.0, "fallback": true}'
[ -f $O/track.json ] || { python eval_ball_detector.py --reuse $O/eval_$W1.json $O/eval_$W2.json \
    --track 'v3P=.cache/v3m_balls_{window}.parquet' --track-params "v3P=$P3" \
    --sealed $O/track.json.part > $O/track.log 2>&1 && mv $O/track.json.part $O/track.json; } || exit 1
echo tracked
[ -f $O/tiles_key/chunk_01.json ] || python build_ball_verify.py --v5 --scores $O/eval_$W1.json $O/eval_$W2.json $O/track.json \
    --out $O/tiles --key-out $O/tiles_key --rows-per-chunk 120 --decoy-rows 6 --seed 52 > $O/tiles.log 2>&1 || exit 1
echo tiles-done
