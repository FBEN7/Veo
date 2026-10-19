#!/bin/sh
cd /home/user/Veo
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad
# existing windows (detections already there): training M4/M6/M8, report M2/M10
python build_reid_crops.py st2_118576_1st_f030000 st2_118578_1st_f030000 st2_128058_1st_f030000 st2_117093_1st_f015000 st2_132877_1st_f015000 > $S/crops_existing.log 2>&1
until [ -f $S/fetch_reid.done ]; do sleep 30; done
python build_reid_crops.py st2_117092_1st_f030000 st2_118577_1st_f030000 st2_118576_2nd_f015000 st2_118578_2nd_f015000 st2_128058_2nd_f015000 > $S/crops_new.log 2>&1
echo done > $S/crops_reid.done
