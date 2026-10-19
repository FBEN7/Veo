#!/bin/sh
cd /home/user/Veo
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad
for w in st2_132877_1st_f015000 st2_117093_1st_f015000 st2_118575_1st_f015000; do
  python $S/reid_report.py $w imagenet_m3 .cache/player_reid_imagenet_s0.pt 0.7 0.85
  python $S/reid_report.py $w imagenet_broadcast .cache/player_reid_imagenet_s0.pt 0.4 0.7
  python $S/reid_report.py $w trained_hinge_m3 .cache/player_reid_st2.pt 0.9 0.97
done > $S/report_runs.log 2> $S/report_runs.err
echo done > $S/report_runs.done
