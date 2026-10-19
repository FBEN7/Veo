#!/bin/sh
cd /home/user/Veo
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball
for m in 118575 118576 118577 118578 128058 117092; do
  python build_ball_patches.py --match $m --half 1 --out .cache/soccertrack_v2/ball_patches >> $S/extract.log 2>&1
done
echo done > $S/extract.done
