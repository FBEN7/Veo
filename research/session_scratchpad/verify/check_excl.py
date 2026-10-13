import sys, glob, pickle
sys.path.insert(0, "/home/user/Veo")
from pathlib import Path
import train_player_reid as T

D = "/home/user/Veo/data/soccertrack_v2/"
C = "/home/user/Veo/.cache/soccertrack_v2/"
train_paths = sorted(glob.glob(C + "reid_train/*.pkl"))
val_paths = sorted(glob.glob(C + "reid_val/*.pkl"))
report_paths = sorted(glob.glob(C + "reid_report/*.pkl"))
print("train", [Path(p).name for p in train_paths])
print("val", [Path(p).name for p in val_paths])

# key types
b = pickle.loads(Path(train_paths[0]).read_bytes())
print("keys", list(b.keys()), "match", b["match"], "window", b["window"])
k = next(iter(b["people"]))
print("pid type", type(k), repr(k))

gts = {"M3": D+"st2_118575_1st_f015000_gt.parquet",
       "M2": D+"st2_117093_1st_f015000_gt.parquet",
       "M10": D+"st2_132877_1st_f015000_gt.parquet"}
ids = {m: T.excluded_ids([p]) for m, p in gts.items()}
print({m: len(v) for m, v in ids.items()})
print("any gt id sample", list(ids["M2"])[:3])

def run(excl, valp):
    drop = T.excluded_ids(excl)
    train = T.load_crops(train_paths, drop)
    val = T.load_crops(valp, pool=False)
    seen = {p for c in train for p in c}
    valset = {p for c in val for p in c}
    print(f"   drop={len(drop)} seen={len(seen)} seen&drop={len(seen & drop)} seen&val={len(seen & valset)}")
    for m, s in ids.items():
        print(f"   seen & GT {m}: {len(seen & s)}")
    return seen

print("== full exclude (3 tables)")
run(list(gts.values()), val_paths)
print("== exclude only M3 table")
run([gts["M3"]], val_paths)
print("== no exclude at all")
run([], val_paths)

# Shared players per training match vs report windows
allpeople = {}
for p in train_paths:
    bb = pickle.loads(Path(p).read_bytes())
    allpeople.setdefault(bb["match"], set()).update(k for k, v in bb["people"].items() if v["role"] == "player")
for m, s in allpeople.items():
    print(m, {k: len(s & v) for k, v in ids.items()})
