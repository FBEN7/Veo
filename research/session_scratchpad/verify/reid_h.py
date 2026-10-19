import sys, glob, pickle
sys.path.insert(0, "/home/user/Veo")
import numpy as np
from pathlib import Path
from train_player_reid import load_crops, excluded_ids, K
root = Path("/home/user/Veo/data/soccertrack_v2")
ex = excluded_ids([root/f"st2_{m}_1st_f015000_gt.parquet" for m in ("117093","118575","132877")])
print("excluded", len(ex))
paths = sorted(glob.glob("/home/user/Veo/.cache/soccertrack_v2/reid_train/*.pkl"))
clips = load_crops(paths, ex)
hs=[]; people=0
for c in clips:
    for pid,(t,crops) in c.items():
        people+=1
        hs += [cr.shape[0] for cr in crops]
hs=np.array(hs)
print("people", people, "crops", len(hs))
print("pct", {p: float(np.percentile(hs,p)) for p in (0,1,5,10,25,50,75,90,95,99,100)})
print("frac<25", (hs<25).mean(), "frac>85", (hs>85).mean(), "frac>70", (hs>70).mean())
# also stored detection h
hh=[]
for p in paths:
    b=pickle.loads(Path(p).read_bytes())
    for pid,pp in b["people"].items():
        if pp["role"]!="player" or pid in ex: continue
        hh += [(pid, b["match"], h) for h in pp["h"]]
# per-match breakdown
import collections
bym=collections.defaultdict(list)
for c,p in zip(clips, []): pass
for p in paths:
    b=pickle.loads(Path(p).read_bytes())
    hs2=[cr.shape[0] for pid,pp in b["people"].items() if pp["role"]=="player" and pid not in ex for cr in pp["crops"]]
    print(Path(p).name, len(hs2), np.percentile(hs2,[5,50,95]) if hs2 else None, max(hs2) if hs2 else None)
print("==== GSR-half exclusion")
g = Path("/home/user/Veo/.cache/soccertrack_v2/gsr")
ex2 = excluded_ids([g/f"{m}_1st.parquet" for m in ("117093","118575","132877")])
print("excluded", len(ex2))
for label, paths2 in (("all", paths), ("noM5", [p for p in paths if "118577" not in p])):
    clips = load_crops(paths2, ex2)
    hs=np.array([cr.shape[0] for c in clips for pid,(t,crops) in c.items() for cr in crops])
    ids=set(pid for c in clips for pid in c)
    print(label, "distinct people", len(ids), "crops", len(hs))
    print("pct", {p: round(float(np.percentile(hs,p)),1) for p in (0,1,5,10,50,90,95,99,100)})
    print("frac<25", round((hs<25).mean(),4), "frac>85", round((hs>85).mean(),4), "frac>70", round((hs>70).mean(),4))
