#!/bin/sh
# M1 (117092, never used by any ball model or event rule): the clean test set for
# the events work -- the 4 non-overlapping 3-minute windows with the most passes
# and carries (labels counted, nothing else read): 1st half from frames 1500 and
# 24000, 2nd half from 0 and 27000. Per window: cut, players and teams as for the
# other windows, v3 dense at stride 1 and its spot marks, v2 dense at stride 2.
# Each step is skipped when done. No events are run or scored here.
set -u
cd /home/user/Veo
S=research/session_scratchpad/ball/events_m1
P=$(python3 /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/find_proxy.py)
[ -n "$P" ] && export HTTPS_PROXY=$P https_proxy=$P
for spec in "1 1500" "1 24000" "2 0" "2 27000"; do
  set -- $spec; H=$1; F=$2
  HN=$([ $H = 1 ] && echo 1st || echo 2nd)
  W=st2_117092_${HN}_f$(printf %06d $F)
  [ -f data/soccertrack_v2/$W.json ] || python fetch_soccertrack_v2.py --match 117092 --half $H \
      --start $(python3 -c "print($F / 1500)") --minutes 3 > $S/fetch_$W.log 2>&1 || exit 1
  [ -f output_$W/tracks.parquet ] || python eval_soccertrack_v2.py data/soccertrack_v2/$W.mp4 \
      --out output_$W --run --model yolov8s.pt --stride 2 --pitch-margin 50 > $S/det_$W.log 2>&1 || exit 1
  [ -f output_$W/events_v1/tracks_teams.parquet ] || python research/session_scratchpad/events_run.py $W events_v1 \
      > $S/ev1_$W.log 2>&1 || exit 1
  [ -f .cache/v3_balls_$W.parquet.meta.json ] || python research/session_scratchpad/ball/v5p2/dense_a2.py $W > $S/v3_$W.log 2>&1 || exit 1
  [ -f .cache/v3m_balls_$W.parquet.meta.json ] || python research/session_scratchpad/ball/v5p2/marks.py $W > $S/marks_$W.log 2>&1 || exit 1
  [ -f .cache/v2_balls_$W.parquet.meta.json ] || python research/session_scratchpad/ball/events_m7m9/dense_v2.py $W > $S/v2_$W.log 2>&1 || exit 1
  echo "$W ready"
done
echo m1-ready
