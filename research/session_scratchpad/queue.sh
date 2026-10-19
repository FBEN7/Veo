#!/bin/sh
# Fetch tuning windows (train) and a second held-out window (validation), then detect on each.
cd /home/user/Veo
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad
for spec in "118576 20 2" "118578 20 2" "128058 20 2" "132877 10 3"; do
  set -- $spec
  python fetch_soccertrack_v2.py --match $1 --half 1 --start $2 --minutes $3 > $S/fetch_$1.log 2>&1
done
for w in st2_118576_1st_f030000 st2_118578_1st_f030000 st2_128058_1st_f030000 st2_132877_1st_f015000; do
  python eval_soccertrack_v2.py data/soccertrack_v2/$w.mp4 --out output_$w --run --model yolov8s.pt --stride 2 --pitch-margin 10 > $S/run_$w.log 2>&1
done
echo QUEUE DONE > $S/queue.done
