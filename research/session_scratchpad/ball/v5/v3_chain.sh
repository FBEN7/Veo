#!/bin/sh
# v3: cut, label and combine the strips, smoke-test the trainer. Restart-proof:
# each step skipped when its output exists (cut resumes per window).
cd /home/user/Veo
V=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5
O=.cache/soccertrack_v2
python build_verified_strips.py cut || exit 1
[ -f $O/ball_strips_verified/verified_codes.npy ] || python build_verified_strips.py label || exit 1
[ -f $O/ball_strips_v3/labels_meta.json ] || python build_verified_strips.py combine || exit 1
echo strips-done
[ -f $V/smoke_v3.20.pt ] || python train_ball_v3.py --init v1 --smoke $V/smoke_v3 || exit 1
echo smoke-done
