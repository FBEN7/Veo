#!/bin/bash
# Fix 2 on real data, through the CLI: a clean --window run, and the same run
# killed (SIGKILL) once its partial parquet holds rows, then run again.
set -u
cd /home/user/Veo
D=/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/c_fix
W=st2_118575_1st_f015000
ARGS="--window $W --frames 15000 15800 --flush-every 3"
rm -rf $D/win_clean $D/win_crash
echo "== clean"
python build_ball_strips.py $ARGS --out $D/win_clean
echo "== crash"
python build_ball_strips.py $ARGS --out $D/win_crash > $D/crash1.log 2>&1 &
PID=$!
PART=$D/win_crash/strips_118575_1_${W}_15000-15800.partial.parquet
for i in $(seq 1 600); do
  if [ -f $PART ]; then
    N=$(python -c "import pandas as pd; print(len(pd.read_parquet('$PART')))" 2>/dev/null || echo 0)
    if [ "$N" -ge 6 ]; then kill -9 $PID; echo "killed with $N rows saved"; break; fi
  fi
  sleep 0.2
done
wait $PID 2>/dev/null
cat $D/crash1.log
ls $D/win_crash | grep -c jpg | sed 's/^/jpgs on disk after the kill: /'
ls $D/win_crash/strips.parquet 2>/dev/null || echo "no strips.parquet after the kill"
echo "== resume"
python build_ball_strips.py $ARGS --out $D/win_crash
echo "== compare"
python - <<EOF
import pandas as pd
from pathlib import Path
a, b = Path("$D/win_clean"), Path("$D/win_crash")
ta, tb = pd.read_parquet(a / "strips.parquet"), pd.read_parquet(b / "strips.parquet")
pd.testing.assert_frame_equal(ta, tb)
fa = sorted(p.name for p in a.glob("*.jpg")); fb = sorted(p.name for p in b.glob("*.jpg"))
assert fa == fb == sorted(ta.file)
same = all((a / f).read_bytes() == (b / f).read_bytes() for f in fa)
left = sorted(p.name for p in b.iterdir() if "partial" in p.name or p.suffix == ".tmp")
print(f"strips.parquet equal ({len(ta)} rows), {len(fa)} JPEGs byte-identical: {same}; leftovers: {left}")
assert same and not left
EOF
