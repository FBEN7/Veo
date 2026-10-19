#!/bin/sh
cd /home/user/Veo
for pair in "118575" "118576" "118577" "128058"; do
  python build_ball_strips.py --match $pair --half 1 --out .cache/soccertrack_v2/ball_strips > /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/strips_${pair}_1.log 2>&1 &
  python build_ball_strips.py --match $pair --half 2 --out .cache/soccertrack_v2/ball_strips > /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/strips_${pair}_2.log 2>&1 &
  wait
done
python label_ball_strips.py --strips .cache/soccertrack_v2/ball_strips --sigma > /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/sigma.log 2>&1
python label_ball_strips.py --strips .cache/soccertrack_v2/ball_strips --compare-window st2_118577_1st_f030000 --strips-a .cache/soccertrack_v2/ball_strips > /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/compare.log 2>&1
python label_ball_strips.py --strips .cache/soccertrack_v2/ball_strips --coco --holdout 300 > /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/labels_v2.log 2>&1
python label_ball_strips.py --strips .cache/soccertrack_v2/ball_strips --montage /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v2_montage.png --montage-n 12 > /dev/null 2>&1
echo done > /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/chain_v2.done
