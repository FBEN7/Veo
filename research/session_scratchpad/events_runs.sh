#!/bin/sh
cd /home/user/Veo
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad
for w in st2_118575_1st_f015000 st2_118576_1st_f030000 st2_118578_1st_f030000 st2_128058_1st_f030000 st2_117093_1st_f015000 st2_132877_1st_f015000; do
  python $S/events_run.py $w events_v1 > $S/events_$w.log 2>&1
done
echo done > $S/events_runs.done
