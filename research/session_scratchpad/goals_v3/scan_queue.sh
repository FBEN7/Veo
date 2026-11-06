#!/bin/sh
# The 16 halves without a 1-fps formation scan, in two lanes of 2 threads
# (shots_goals/formation_scan.py, resumable: a restart continues from the
# last chunk; the video's signed URL expiring stops a try, the next resumes).
#   sh scan_queue.sh LANE   (LANE 0 or 1)
set -u
cd /home/user/veo
S=research/session_scratchpad/shots_goals
L=research/session_scratchpad/goals_v3/logs
mkdir -p $L
export PYTHONPATH=/home/user/veo OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
HALVES="117093:1 117093:2 118575:1 118575:2 118576:1 118576:2 118577:1 118578:1 118578:2 128057:1 128057:2 128058:1 128058:2 132831:1 132877:1 132877:2"
i=0
for mh in $HALVES; do
  if [ $((i % 2)) -eq "$1" ]; then
    m=${mh%:*}; h=${mh#*:}; H=$([ "$h" = 1 ] && echo 1st || echo 2nd)
    for t in 1 2 3 4 5 6 7 8 9 10 11 12; do
      if python -c "import json,sys; sys.exit(0 if json.load(open('.cache/formation_scan/${m}_$H.json'))['complete'] else 1)" 2>/dev/null; then
        break
      fi
      nice -n 10 python $S/formation_scan.py "$m" "$h" --threads 2 >> $L/scan_${m}_$h.log 2>&1
    done
    echo "$(date -u +%H:%M) ${m}_$H done or stopped" >> $L/queue_$1.log
  fi
  i=$((i + 1))
done
