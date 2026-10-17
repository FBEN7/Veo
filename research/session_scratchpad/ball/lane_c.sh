#!/bin/sh
# Second dense lane for V-new, from the last window back, once the raw
# eval lane is done (cores free). Windows are locked per process.
cd /home/user/Veo
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball
V=$S/vnew
until grep -qE "lane-b-done|failed" $V/lanes.log 2>/dev/null; do sleep 60; done
for w in st2_132877_2nd_f045000 st2_132877_2nd_f015000 st2_132877_1st_f045000 st2_117093_2nd_f045000 st2_117093_2nd_f015000; do
  grep -q merged $V/dense_$w.log 2>/dev/null && continue
  python $S/dense_chunked.py $w >> $V/dense_$w.log 2>&1 || echo "lane-c dense $w failed" >> $V/lanes.log
done
echo lane-c-done >> $V/lanes.log
