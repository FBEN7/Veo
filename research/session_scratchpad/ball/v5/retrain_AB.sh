#!/bin/sh
# Step 1c: the tile classifier retrained on M6's judged tiles plus the judged
# tiles of the 6 other seen matches (spot check + 400 picked). Leave-one-match-out
# CV of two settings: base (as model_A) and fix (no tracker flag -- the tracker
# was not run for 4 of the matches -- and brightness augmentation 0.6-1.4: 3
# matches are darker than M6). Restart-proof: steps skipped when their output
# exists; CV folds kept per match.
cd /home/user/Veo
V=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5
O=.cache/soccertrack_v2/ball_verified
[ -f $O/B/judged_train.parquet ] || python spot_check_tiles.py collect --key $V/train_B_key/key.json \
    --judged $V/train_B_judgements --out $O/B/judged_train.parquet || exit 1
[ -f $O/clf/crops_Bj.npz ] || python train_tile_classifier.py label --crops $O/clf/crops_B.npz \
    --judged $O/B/judged_spot.parquet $O/B/judged_train.parquet --out $O/clf/crops_Bj.npz || exit 1
C="$O/clf/crops_A.npz $O/clf/crops_Bj.npz"
FIX="--drop-meta T --bright 0.6 1.4"
G1="117093 118575 118576 118578"; G2="118577 128058 132877"
if [ ! -f $O/clf/cv_AB.json ] || [ ! -f $O/clf/cv_AB_base.json ]; then
  python train_tile_classifier.py cv --crops $C --by match --threads 1 $FIX --out $O/clf/cv_AB.json --groups $G1 > $V/cv_fix_1.log 2>&1 &
  python train_tile_classifier.py cv --crops $C --by match --threads 1 $FIX --out $O/clf/cv_AB.json --groups $G2 > $V/cv_fix_2.log 2>&1 &
  python train_tile_classifier.py cv --crops $C --by match --threads 1 --out $O/clf/cv_AB_base.json --groups $G1 > $V/cv_base_1.log 2>&1 &
  python train_tile_classifier.py cv --crops $C --by match --threads 1 --out $O/clf/cv_AB_base.json --groups $G2 > $V/cv_base_2.log 2>&1 &
  wait
  python train_tile_classifier.py cv --crops $C --by match $FIX --out $O/clf/cv_AB.json || exit 1
  python train_tile_classifier.py cv --crops $C --by match --out $O/clf/cv_AB_base.json || exit 1
fi
echo cv-done
[ -f $O/clf/model_AB.pt ] || { python train_tile_classifier.py fit --crops $C $FIX --threads 4 --out $O/clf/model_AB.pt.part \
    && mv $O/clf/model_AB.pt.part $O/clf/model_AB.pt; } || exit 1
echo retrain-done
