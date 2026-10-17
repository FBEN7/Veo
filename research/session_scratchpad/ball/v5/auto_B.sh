#!/bin/sh
# Step 1b: once every candidate job is done, label the other 14 seen windows
# with the tile classifier trained on M6's judged tiles. Restart-proof: each
# step is skipped when its output exists.
cd /home/user/Veo
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball
V=$S/v5; O=.cache/soccertrack_v2/ball_verified
mkdir -p $O/B
until [ $(ls $V/cand_*_*_*.json 2>/dev/null | grep -v part | wc -l) -ge 56 ]; do sleep 120; done
until [ -f $O/clf/model_A.pt ]; do sleep 60; done
SC="$V/cand_*_*_*.json $S/vnew/raw_*.json $S/Vnew_sealed.json $S/V.json $V/track_Vold.json"
[ -f $O/B/candidates.parquet ] || python build_ball_verify.py --scores $SC --windows st2_117093_1st_f015000 st2_117093_1st_f045000 st2_117093_2nd_f015000 st2_117093_2nd_f045000 st2_118575_1st_f015000 st2_118576_1st_f030000 st2_118576_2nd_f015000 st2_118577_1st_f030000 st2_128058_1st_f030000 st2_128058_2nd_f015000 st2_132877_1st_f015000 st2_132877_1st_f045000 st2_132877_2nd_f015000 st2_132877_2nd_f045000 --tiles-out $O/B/candidates.parquet || exit 1
[ -f $O/clf/crops_B.npz ] || python train_tile_classifier.py crops --tiles $O/B/candidates.parquet --out $O/clf/crops_B.npz || exit 1
python train_tile_classifier.py apply --model $O/clf/model_A.pt --crops $O/clf/crops_B.npz --out $O/B/auto_labels.parquet || exit 1
echo auto-B-done
