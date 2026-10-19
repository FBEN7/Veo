#!/bin/sh
# Shot gate (protocol.json SHOT GATE), all 26 attempts, resumable: for each BAS
# SHOT/GOAL of 128057 half 1 and 132831 half 2 a clip around the label
# (data/shot_gate/soccertrack_v2/), v3 at stride 1 on [-1 s, +2.2 s] into
# gate_v3/<match>_<half>_<frame>.parquet (an attempt whose sidecar covers its
# range is skipped), then the score with the reproducibility check and the
# decision into ball_gate.json. See ball_gate.py for the readings.
#
# M7 / M9 are sealed: export VEO_UNSEAL (the sha256 of
# .cache/soccertrack_v2/ball_verified/v5p2_registration.json) first.
# 132831 half 2 is a GOALS test half: its attempts are refused until
# frozen_goals.json (+ .sha256) exists here, unless --before-goals-freeze is
# passed (it is handed on to ball_gate.py and written into the outputs).
# Expected cost: about 4 min an attempt (clip fetch, then v3 on 81 frames with
# 2 threads; GATE_THREADS to change), about 2 h for 26: run it in the background.
#
#     VEO_UNSEAL=... sh research/session_scratchpad/shots_goals/gate_driver.sh [--before-goals-freeze]
set -u
cd /home/user/Veo
S=research/session_scratchpad/shots_goals
export PYTHONPATH=/home/user/Veo
[ -n "${VEO_UNSEAL:-}" ] || { echo "VEO_UNSEAL is not set: M7 / M9 stay sealed (src/unseal.py)"; exit 1; }
mkdir -p $S/gate_v3
exec 9>$S/gate_v3/.lock
flock -n 9 || { echo "another gate_driver.sh holds $S/gate_v3/.lock"; exit 1; }
python $S/ball_gate.py detect --threads "${GATE_THREADS:-2}" "$@" >> $S/gate_detect.log 2>&1 \
  || { echo "detect stopped (see $S/gate_detect.log); rerun to resume"; exit 1; }
echo "gate detections done"
python $S/ball_gate.py score "$@" > $S/gate_score.log 2>&1 \
  || { echo "score failed (see $S/gate_score.log)"; exit 1; }
tail -1 $S/gate_score.log
echo gate-done
