#!/bin/sh
# One relaunch of the M1 preparation within a 2-hour task cap: player detection
# (not resumable, ~1.5 h a window) runs first, for the next window without it,
# with driver.sh's own commands; then driver.sh, whose other steps resume.
set -u
cd /home/user/Veo
S=research/session_scratchpad/ball/events_m1
P=$(python3 /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/find_proxy.py)
[ -n "$P" ] && export HTTPS_PROXY=$P https_proxy=$P
for spec in "1 1500" "1 24000" "2 0" "2 27000"; do
  set -- $spec; H=$1; F=$2
  HN=$([ $H = 1 ] && echo 1st || echo 2nd)
  W=st2_117092_${HN}_f$(printf %06d $F)
  [ -f output_$W/tracks.parquet ] && continue
  [ -f data/soccertrack_v2/$W.json ] || python fetch_soccertrack_v2.py --match 117092 --half $H \
      --start $(python3 -c "print($F / 1500)") --minutes 3 > $S/fetch_$W.log 2>&1 || exit 1
  python eval_soccertrack_v2.py data/soccertrack_v2/$W.mp4 \
      --out output_$W --run --model yolov8s.pt --stride 2 --pitch-margin 50 > $S/det_$W.log 2>&1 || exit 1
  break
done
sh $S/driver.sh
