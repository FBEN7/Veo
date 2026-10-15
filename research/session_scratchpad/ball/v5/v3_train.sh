#!/bin/sh
# v3: train both runs, evaluate them on M6's second half (the registered selection
# set, amendment 1), score. Restart-proof: each step skipped when its output exists;
# training resumes from <out>.resume.pt.
cd /home/user/Veo
V=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball
[ -f .cache/ball_v3a.2000.pt.provenance.json ] || python train_ball_v3.py --init v2 --out .cache/ball_v3a >> $V/train_v3a.log 2>&1 || exit 1
echo v3a-done
[ -f .cache/ball_v3b.2000.pt.provenance.json ] || python train_ball_v3.py --init v1 --out .cache/ball_v3b >> $V/train_v3b.log 2>&1 || exit 1
echo v3b-done
for W in st2_118578_2nd_f015000 st2_118578_2nd_f037500; do
  [ -f $V/v3eval_a_$W.json ] || python eval_ball_detector.py --windows $W --no-coco \
    --v2 .cache/ball_v3a.1000.pt .cache/ball_v3a.1500.pt .cache/ball_v3a.2000.pt .cache/ball_v2.4000.pt \
    --save $V/v3eval_a_$W.json.part > $V/v3eval_a_$W.log 2>&1 && mv $V/v3eval_a_$W.json.part $V/v3eval_a_$W.json || exit 1
  [ -f $V/v3eval_b_$W.json ] || python eval_ball_detector.py --windows $W --no-coco --allow-seen \
    --v2 .cache/ball_v3b.1000.pt .cache/ball_v3b.1500.pt .cache/ball_v3b.2000.pt \
    --save $V/v3eval_b_$W.json.part > $V/v3eval_b_$W.log 2>&1 && mv $V/v3eval_b_$W.json.part $V/v3eval_b_$W.json || exit 1
done
echo eval-done
python score_verified_top1.py --scores $S/U_a.json $S/U_b.json $V/v3eval_b_*.json $V/v3eval_a_*.json \
  --windows st2_118578_2nd_f015000 st2_118578_2nd_f037500 \
  --detectors ball_v3a.1000.pt ball_v3a.1500.pt ball_v3a.2000.pt ball_v3b.1000.pt ball_v3b.1500.pt ball_v3b.2000.pt ball_v2.4000.pt ball_heatmap.3000.pt \
  --out .cache/soccertrack_v2/ball_verified/v3_selection_scores.json > $V/v3_score.log 2>&1 || exit 1
echo v3-all-done
