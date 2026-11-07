#!/bin/sh
# The registered goals test: scan the four test halves (resumable), then score once.
# Rerun after a restart or the 2-h cap: finished halves are skipped.
set -u
cd /home/user/Veo
S=research/session_scratchpad/shots_goals
for MH in "117092 1" "117092 2" "132831 2" "118577 2"; do
  set -- $MH
  sh $S/scan_driver.sh $1 $2 || exit 1
done
[ -f $S/goals_test.json ] || PYTHONPATH=/home/user/Veo python $S/score_goals_test.py > $S/goals_test.log 2>&1
echo goals-test-done
