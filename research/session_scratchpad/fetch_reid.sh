#!/bin/sh
cd /home/user/Veo
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad
for spec in "117092 1 20" "118577 1 20" "118576 2 10" "118578 2 10" "128058 2 10"; do
  set -- $spec
  python fetch_soccertrack_v2.py --match $1 --half $2 --start $3 --minutes 3 > $S/fetchreid_$1_$2.log 2>&1
done
echo done > $S/fetch_reid.done
