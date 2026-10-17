#!/bin/sh
# Descriptive events on M7 / M9 (after protocol v5 part 2): players and teams as
# for the old windows (eval_soccertrack_v2.py --run, then events_run.py), v2's
# dense rows at stride 2, then the frozen events_v3 rules with v2, v3 raw and v3
# with the spot marks dropped. Each step is skipped when done.
set -u
cd /home/user/Veo
S=research/session_scratchpad/ball/events_m7m9
W7=st2_128057_1st_f030000; W9=st2_132831_2nd_f030000
for W in $W7 $W9; do
  [ -f output_$W/tracks.parquet ] || python eval_soccertrack_v2.py data/soccertrack_v2/$W.mp4 \
      --out output_$W --run --model yolov8s.pt --stride 2 --pitch-margin 50 > $S/det_$W.log 2>&1 || exit 1
  echo "$W players"
  [ -f output_$W/events_v1/tracks_teams.parquet ] || python research/session_scratchpad/events_run.py $W events_v1 \
      > $S/ev1_$W.log 2>&1 || exit 1
  [ -f .cache/v2_balls_$W.parquet.meta.json ] || python $S/dense_v2.py $W > $S/v2_$W.log 2>&1 || exit 1
  for c in v2 v3 v3m; do
    case $c in
      v2) R=.cache/v2_balls_$W.parquet; C=.cache/ball_v2.4000.pt.calib.json ;;
      v3) R=.cache/v3_balls_$W.parquet; C=.cache/ball_v3b.1500.pt.calib.json ;;
      v3m) R=.cache/v3m_balls_$W.parquet; C=.cache/ball_v3b.1500.pt.calib.json ;;
    esac
    [ -f output_$W/ev_$c/applied.json ] || python $S/apply.py $W ev_$c $R $C >> $S/apply.log 2>&1 || exit 1
  done
  echo "$W events"
done
python score_events_paired.py --windows $W7 $W9 --configs events_v1 ev_v2 ev_v3 ev_v3m --json $S/scores.json \
    > $S/scores.log 2>&1 || exit 1
echo events-done
