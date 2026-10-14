#!/bin/sh
cd /home/user/Veo
O=.cache/soccertrack_v2/ball_verified/clf
[ -f $O/model_A.pt ] || { python train_tile_classifier.py fit --crops $O/crops_A.npz --out $O/model_A.pt.part && mv $O/model_A.pt.part $O/model_A.pt; }
echo fit-done
