#!/bin/sh
# Step 1 (verified labels): detector candidates at every seen window's measured
# frames, one job per window x frame kind (eval --frames KIND) so a worker restart
# loses at most one short job. Worker: claims jobs by mkdir lock (pid inside; a
# dead holder's lock is taken over), skips finished ones; --save output is written
# to .part then renamed.
cd /home/user/Veo
V=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5
A="--v2 .cache/ball_v2.4000.pt --heatmap .cache/ball_heatmap.3000.pt --allow-seen"
H1="--heatmap .cache/ball_heatmap.3000.pt --no-coco"
V2="--v2 .cache/ball_v2.4000.pt --no-coco"
WA="st2_118575_1st_f015000 st2_118576_1st_f030000 st2_118577_1st_f030000 st2_128058_1st_f030000 st2_118576_2nd_f015000 st2_128058_2nd_f015000"
W1="st2_117093_1st_f045000 st2_117093_2nd_f015000 st2_117093_2nd_f045000 st2_132877_1st_f045000 st2_132877_2nd_f015000 st2_132877_2nd_f045000"
W2="st2_117093_1st_f015000 st2_132877_1st_f015000"
JOBS=""
for f in passes actions random placebo; do
  for w in $WA; do JOBS="$JOBS all:$w:$f"; done
  for w in $W1; do JOBS="$JOBS v1:$w:$f"; done
  for w in $W2; do JOBS="$JOBS v2:$w:$f"; done
done
for j in $JOBS; do
  k=${j%%:*}; rest=${j#*:}; w=${rest%%:*}; f=${rest#*:}
  out=$V/cand_${k}_${w}_$f.json; lock=$V/lock_${k}_${w}_$f
  [ -f $out ] && continue
  if ! mkdir $lock 2>/dev/null; then
    pid=$(cat $lock/pid 2>/dev/null)
    if [ -n "$pid" ] && grep -q cand_queue /proc/$pid/cmdline 2>/dev/null; then continue; fi
  fi
  echo $$ > $lock/pid
  case $k in all) args=$A;; v1) args=$H1;; v2) args=$V2;; esac
  echo "$(date -u +%H:%M) start $k $w $f" >> $V/queue.log
  if python eval_ball_detector.py $args --windows $w --frames $f --threads 2 --save $out.part.json > $V/cand_${k}_${w}_$f.log 2>&1; then
    mv $out.part.json $out; echo "$(date -u +%H:%M) done $k $w $f" >> $V/queue.log
  else
    echo "$(date -u +%H:%M) FAILED $k $w $f" >> $V/queue.log
  fi
  rm -rf $lock
done
echo "$(date -u +%H:%M) worker $$ finished" >> $V/queue.log
