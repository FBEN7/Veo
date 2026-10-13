#!/bin/sh
cd /home/user/Veo
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball
P=.cache/soccertrack_v2/ball_patches
until grep -q "128058 half 1" $S/extract.log; do sleep 30; done
for p in $(pgrep -f '[e]xtract.sh'); do kill $p; done
for p in $(pgrep -f '[b]uild_ball_patches.py --match 117092'); do kill $p; done
python build_ball_labels.py --patches $P --windows st2_118575_1st_f015000 st2_118576_1st_f030000 st2_118577_1st_f030000 st2_118578_1st_f030000 st2_128058_1st_f030000 > $S/labels.log 2>&1 || exit 1
python train_ball_heatmap.py --patches $P --out .cache/ball_heatmap > $S/train.log 2>&1 || exit 1
python eval_ball_detector.py --no-coco --heatmap .cache/ball_heatmap.1500.pt .cache/ball_heatmap.2250.pt .cache/ball_heatmap.3000.pt --windows st2_118576_2nd_f015000 st2_118578_2nd_f015000 st2_128058_2nd_f015000 --save $S/heatmap_T.json > $S/heatmap_T.log 2>&1
echo done > $S/chain.done
