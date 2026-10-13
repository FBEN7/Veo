#!/bin/sh
cd /home/user/Veo
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad
R=.cache/soccertrack_v2
python build_reid_crops.py st2_118575_1st_f015000 --out $R/reid_val > $S/crops2_val.log 2>&1
python build_reid_crops.py st2_117093_1st_f015000 st2_132877_1st_f015000 --out $R/reid_report > $S/crops2_report.log 2>&1
python build_reid_crops.py st2_118576_1st_f030000 st2_118578_1st_f030000 st2_128058_1st_f030000 --out $R/reid_train > $S/crops2_train_a.log 2>&1
until [ -f $S/fetch_reid.done ]; do sleep 30; done
python build_reid_crops.py st2_117092_1st_f030000 st2_118577_1st_f030000 st2_118576_2nd_f015000 st2_118578_2nd_f015000 st2_128058_2nd_f015000 --out $R/reid_train > $S/crops2_train_b.log 2>&1
echo done > $S/crops_reid2.done
