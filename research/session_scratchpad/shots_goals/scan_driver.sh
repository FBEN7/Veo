#!/bin/sh
# One half's 1-fps formation scan, resumable (chunks under .cache/formation_scan/<m>_<h>.parts).
# The video's signed URL expires during a long read (HTTP 403): each retry reopens it and
# continues from the last chunk; up to 12 tries. CPU is shared: nice, 2 threads.
#   sh scan_driver.sh MATCH HALF
set -u
cd /home/user/Veo
S=research/session_scratchpad/shots_goals
H=$([ "$2" = 1 ] && echo 1st || echo 2nd)
export PYTHONPATH=/home/user/Veo OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  nice -n 10 python $S/formation_scan.py "$1" "$2" --threads 2 >> $S/scan_$1_$2.log 2>&1
  if python -c "import json,sys; sys.exit(0 if json.load(open('.cache/formation_scan/$1_$H.json'))['complete'] else 1)" 2>/dev/null; then
    echo "scan $1 $2 complete"; exit 0
  fi
  echo "scan $1 $2: try $i stopped, resuming" >> $S/scan_$1_$2.log
done
echo "scan $1 $2 not complete after 12 tries"; exit 1
