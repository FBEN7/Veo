#!/bin/sh
# Learned events, training data: v3 dense rows (stride 1) and spot marks for the six
# older windows that have players and teams cached (M2, M3, M4, M6 1st, M8, M10).
# v3 was trained on these matches (--allow-seen): its ball is better here than on
# unseen footage, a bias the M1 test does not share. Each step resumes.
set -u
cd /home/user/Veo
S=research/session_scratchpad/ball/events_learn
for W in st2_117093_1st_f015000 st2_118575_1st_f015000 st2_118576_1st_f030000 st2_118578_1st_f030000 st2_128058_1st_f030000 st2_132877_1st_f015000; do
  [ -f .cache/v3_balls_$W.parquet.meta.json ] || python $S/dense_seen.py $W > $S/v3_$W.log 2>&1 || exit 1
  [ -f .cache/v3m_balls_$W.parquet.meta.json ] || python research/session_scratchpad/ball/v5p2/marks.py $W > $S/marks_$W.log 2>&1 || exit 1
  echo "$W ready"
done
echo train-ready
