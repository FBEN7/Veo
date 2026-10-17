#!/bin/sh
# Protocol v4, V-new: cut, dense (lane A) and sealed raw eval (lane B) in
# parallel, then the tracker on the sealed raw files. Restart-proof: every
# stage skips what is already complete. Prints no measure.
cd /home/user/Veo
S=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball
V=$S/vnew
mkdir -p $V
W="st2_117093_1st_f045000 st2_117093_2nd_f015000 st2_117093_2nd_f045000 st2_132877_1st_f045000 st2_132877_2nd_f015000 st2_132877_2nd_f045000"

for w in $W; do
  [ -f data/soccertrack_v2/$w.json ] && continue
  m=$(echo $w | cut -d_ -f2)
  h=$(echo $w | cut -d_ -f3 | cut -c1)
  f=$(echo $w | cut -d_ -f4 | sed 's/^f0*//')
  min=$((f / 1500))
  echo "cut $w (match $m half $h start $min min)" >> $V/fetch.log
  python fetch_soccertrack_v2.py --match $m --half $h --start $min --minutes 3 >> $V/fetch.log 2>&1 || { echo "cut $w failed" >> $V/fetch.log; exit 1; }
done
echo cut-done >> $V/fetch.log

lane_a() {
  for w in $W; do
    grep -q merged $V/dense_$w.log 2>/dev/null && continue
    python $S/dense_chunked.py $w >> $V/dense_$w.log 2>&1 || { echo "dense $w failed" >> $V/lanes.log; return 1; }
  done
  echo lane-a-done >> $V/lanes.log
}
lane_b() {
  for w in $W; do
    [ -f $V/raw_$w.json ] && continue
    python eval_ball_detector.py --v2 .cache/ball_v2.4000.pt --windows $w --threads 2 \
      --sealed $V/raw_$w.part.json >> $V/raw_$w.log 2>&1 || { echo "raw $w failed" >> $V/lanes.log; return 1; }
    mv $V/raw_$w.part.json $V/raw_$w.json
  done
  echo lane-b-done >> $V/lanes.log
}
lane_a & A=$!
lane_b & B=$!
wait $A; wait $B
for w in $W; do
  grep -q merged $V/dense_$w.log 2>/dev/null && [ -f $V/raw_$w.json ] || { echo "incomplete $w" >> $V/lanes.log; exit 1; }
done
[ -f $S/Vnew_sealed.json ] || python eval_ball_detector.py --reuse $(for w in $W; do printf '%s ' $V/raw_$w.json; done) \
  --track "T=.cache/v2_balls_{window}.parquet" --track-params $S/track_params.json \
  --sealed $S/Vnew_sealed.json > $S/Vnew_sealed.log 2>&1 || { echo "track failed" >> $V/lanes.log; exit 1; }
echo all-done >> $V/lanes.log
