#!/bin/sh
cd /home/user/Veo
O=.cache/soccertrack_v2/ball_verified/clf
[ -f $O/crops_A.npz ] || python train_tile_classifier.py crops --tiles .cache/soccertrack_v2/ball_verified/A/tiles.parquet --out $O/crops_A.npz
python train_tile_classifier.py cv --crops $O/crops_A.npz --out $O/cv_A.json
echo cv-done
