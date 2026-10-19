#!/bin/sh
# After auto_B.sh: render the 100-tile spot check of the automatic labels.
cd /home/user/Veo
V=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5
O=.cache/soccertrack_v2/ball_verified
until [ -f $O/B/auto_labels.parquet ]; do sleep 120; done
[ -f $V/spot_B_key/key.json ] || python spot_check_tiles.py sheets --labels $O/B/auto_labels.parquet --n 100 --seed 0 --out $V/spot_B --key-out $V/spot_B_key || exit 1
echo spot-ready
