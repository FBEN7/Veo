#!/bin/sh
cd /home/user/Veo
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad
G=.cache/soccertrack_v2/gsr; T=.cache/soccertrack_v2/reid_train
TRAIN="$T/st2_117092_1st_f030000.pkl $T/st2_118576_1st_f030000.pkl $T/st2_118576_2nd_f015000.pkl $T/st2_118578_1st_f030000.pkl $T/st2_118578_2nd_f015000.pkl $T/st2_128058_1st_f030000.pkl $T/st2_128058_2nd_f015000.pkl"
EXCL="$G/117093_1st.parquet $G/118575_1st.parquet $G/132877_1st.parquet $G/118577_1st.parquet"
run() { name=$1; shift; python train_player_reid.py "$@" --steps 1000 --crops $TRAIN --val-crops $T/st2_118577_1st_f030000.pkl --exclude $EXCL --out $S/probe_$name.pt > $S/probe_$name.log 2>&1; }
run hard_hinge_3e-4
run soft_1e-4 --soft-margin --lr 1e-4
run all_3e-4 --batch-all
run all_1e-4 --batch-all --lr 1e-4
echo done > $S/probes.done
