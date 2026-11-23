#!/bin/sh
# Shot gate (protocol.json SHOT GATE), all 26 attempts, resumable: for each BAS
# SHOT/GOAL of 128057 half 1 and 132831 half 2 a clip around the label
# (data/shot_gate/soccertrack_v2/), v3 at stride 1 on [-1 s, +2.2 s] into
# gate_v3/<match>_<half>_<frame>.parquet (an attempt whose sidecar is a
# complete run of its planned clip, range and checkpoint is skipped), then the
# score with the reproducibility check and the decision into ball_gate.json.
# See ball_gate.py for the readings.
#
# M7 / M9 are sealed: export VEO_UNSEAL (the sha256 of
# .cache/soccertrack_v2/ball_verified/v5p2_registration.json) first.
# 132831 half 2 is a GOALS test half: its attempts are refused until
# frozen_goals.json and frozen_goals.json.sha256 exist here, agree, and the
# json records the protocol sha256 and T, N and the team rule
# (protocol_notes.json: the goals rule is frozen before the gate reads
# anything of 132831 half 2). There is no override, and the driver takes no
# arguments.
#
# CPU is shared: python (and the detector it starts) runs under nice -n 10
# with OMP / MKL / OpenBLAS limited to GATE_THREADS (default 2) threads, and
# torch gets --threads GATE_THREADS. The clip fetch (fetch_soccertrack_v2.fetch
# -> st.write_clip, libx264) has no thread option, so its encoder picks its own
# thread count; it runs under the same nice.
# Expected cost: about 4 min an attempt (clip fetch, then v3 on 81 frames with
# 2 threads), about 2 h for 26: run it in the background.
#
#     VEO_UNSEAL=... sh research/session_scratchpad/shots_goals/gate_driver.sh
set -u
[ $# -eq 0 ] || { echo "gate_driver.sh takes no arguments (got: $*)"; exit 2; }
cd /home/user/Veo || exit 1
S=research/session_scratchpad/shots_goals
T=${GATE_THREADS:-2}
export PYTHONPATH=/home/user/Veo
export OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T"
[ -n "${VEO_UNSEAL:-}" ] || { echo "VEO_UNSEAL is not set: M7 / M9 stay sealed (src/unseal.py)"; exit 1; }
mkdir -p $S/gate_v3
exec 9>$S/gate_v3/.lock
flock -n 9 || { echo "another gate_driver.sh holds $S/gate_v3/.lock"; exit 1; }
nice -n 10 python $S/ball_gate.py detect --threads "$T" >> $S/gate_detect.log 2>&1 \
  || { echo "detect stopped (see $S/gate_detect.log); rerun to resume"; exit 1; }
echo "gate detections done"
nice -n 10 python $S/ball_gate.py score > $S/gate_score.log 2>&1 \
  || { echo "score failed (see $S/gate_score.log)"; exit 1; }
tail -1 $S/gate_score.log
echo gate-done
