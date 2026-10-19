#!/bin/sh
# Dense v3 (stride 1) on M6's second half, two windows side by side (2 threads each).
cd /home/user/Veo
V=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5
for W in st2_118578_2nd_f015000 st2_118578_2nd_f037500; do
  [ -f .cache/v3_balls_$W.parquet.meta.json ] || python $V/dense_v3.py $W >> $V/dense_v3_$W.log 2>&1 &
done
wait
ls .cache/v3_balls_st2_118578_2nd_f015000.parquet.meta.json .cache/v3_balls_st2_118578_2nd_f037500.parquet.meta.json && echo dense-done
