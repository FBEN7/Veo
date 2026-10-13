#!/bin/sh
cd /home/user/Veo
python eval_ball_detector.py --heatmap .cache/ball_heatmap.3000.pt --windows st2_117093_1st_f015000 st2_132877_1st_f015000 --save /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/V.json > /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/V.log 2>&1
for w in st2_117093_1st_f015000 st2_132877_1st_f015000; do
  python detect_ball_heatmap.py --window $w --weights .cache/ball_heatmap.3000.pt --tau 0.7 --top-k 4 --stride 2 --threads 2 --out .cache/heatmap_balls_$w.parquet > /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/detect_$w.log 2>&1 &
done
wait
echo done > /tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/chain_v.done
