#!/bin/sh
cd /home/user/Veo
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad
until [ -f $S/crops_reid2.done ]; do sleep 30; done
G=.cache/soccertrack_v2/gsr
python train_player_reid.py --crops .cache/soccertrack_v2/reid_train/*.pkl \
  --val-crops .cache/soccertrack_v2/reid_val/*.pkl \
  --exclude $G/117093_1st.parquet $G/118575_1st.parquet $G/132877_1st.parquet \
     data/soccertrack_v2/st2_117093_1st_f015000_gt.parquet data/soccertrack_v2/st2_118575_1st_f015000_gt.parquet data/soccertrack_v2/st2_132877_1st_f015000_gt.parquet \
  --out .cache/player_reid_st2.pt > $S/train_reid.log 2>&1
echo done > $S/train_reid.done
