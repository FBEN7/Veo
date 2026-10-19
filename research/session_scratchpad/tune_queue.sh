#!/bin/sh
# After each training window's pipeline run finishes, run the team sweep on it.
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad
cd /home/user/Veo
for pair in "st2_118576_1st_f030000 M4" "st2_118578_1st_f030000 M6" "st2_128058_1st_f030000 M8"; do
  set -- $pair
  until [ -f output_$1/soccertrack_v2_scores.json ] || grep -q Traceback $S/run_$1.log 2>/dev/null; do sleep 30; done
  python $S/tune.py $1 $2 > $S/tune_$2.log 2>&1
done
echo done > $S/tune_queue.done
