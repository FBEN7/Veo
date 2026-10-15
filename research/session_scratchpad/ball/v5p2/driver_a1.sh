#!/bin/sh
# Protocol v5 part 2, amendment 1: run.sh's steps split into restart-sized jobs
# (worker restarts kill every process every 10-30 min). Same frozen code and
# options: eval_ball_detector.py per window x frame kind x detector (--frames, one
# detector per job), merged by load_scores as before; dense, marks, tracking and
# tiles as in run.sh. Each job is skipped when its output exists. Counts only.
set -u
cd /home/user/Veo
R=research/session_scratchpad/ball/v5p2
O=.cache/soccertrack_v2/ball_verified/v5p2
mkdir -p $O/a1
W1=st2_128057_1st_f030000; W2=st2_132831_2nd_f030000
[ -f data/soccertrack_v2/$W1.json ] && [ -f data/soccertrack_v2/$W2.json ] || { echo "windows missing"; exit 1; }
for W in $W1 $W2; do
  for K in passes actions random placebo; do
    for D in v2 v3 v1 coco; do
      out=$O/a1/eval_${W}_${D}_${K}.json
      [ -f $out ] && continue
      case $D in
        v2) A="--v2 .cache/ball_v2.4000.pt --no-coco" ;;
        v3) A="--v2 .cache/ball_v3b.1500.pt --no-coco" ;;
        v1) A="--heatmap .cache/ball_heatmap.3000.pt --no-coco" ;;
        coco) A="" ;;
      esac
      python eval_ball_detector.py --windows $W $A --frames $K --sealed $out.part > $O/a1/eval_${W}_${D}_${K}.log 2>&1 \
        && mv $out.part $out || exit 1
    done
  done
  [ -f .cache/v3_balls_$W.parquet.meta.json ] || python $R/dense_v5p2.py $W > $O/dense_$W.log 2>&1 || exit 1
  [ -f .cache/v3m_balls_$W.parquet.meta.json ] || python $R/marks.py $W > $O/marks_$W.log 2>&1 || exit 1
done
echo detectors-done
P3='{"tau": 0.1, "top_k": 4, "c_null": 1.0, "c_static": 2.0, "gap": 6, "max_fill_speed": 35.0, "fallback": true}'
[ -f $O/track.json ] || { python eval_ball_detector.py --reuse $O/a1/eval_*.json \
    --track 'v3P=.cache/v3m_balls_{window}.parquet' --track-params "v3P=$P3" \
    --sealed $O/track.json.part > $O/track.log 2>&1 && mv $O/track.json.part $O/track.json; } || exit 1
echo tracked
[ -f $O/tiles_key/chunk_01.json ] || python build_ball_verify.py --v5 --scores $O/a1/eval_*.json $O/track.json \
    --out $O/tiles --key-out $O/tiles_key --rows-per-chunk 120 --decoy-rows 6 --seed 52 > $O/tiles.log 2>&1 || exit 1
echo tiles-done
