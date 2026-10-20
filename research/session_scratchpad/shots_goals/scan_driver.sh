#!/bin/sh
# One half's 1-fps formation scan, resumable (chunks under .cache/formation_scan/<m>_<h>.parts):
# rerun after a restart or the 2-h cap and it continues. CPU is shared: nice, 2 threads.
#   sh scan_driver.sh MATCH HALF
set -u
cd /home/user/Veo
S=research/session_scratchpad/shots_goals
export PYTHONPATH=/home/user/Veo OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
nice -n 10 python $S/formation_scan.py "$1" "$2" --threads 2 >> $S/scan_$1_$2.log 2>&1 \
  && echo "scan $1 $2 complete" || echo "scan $1 $2 stopped (rerun to resume)"
