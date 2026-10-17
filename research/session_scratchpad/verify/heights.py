import glob, pickle, numpy as np, sys
sys.path.insert(0, ".")
from train_player_reid import excluded_ids
G=".cache/soccertrack_v2/gsr"
ex = excluded_ids([f"{G}/117093_1st.parquet",f"{G}/118575_1st.parquet",f"{G}/132877_1st.parquet","data/soccertrack_v2/st2_117093_1st_f015000_gt.parquet","data/soccertrack_v2/st2_118575_1st_f015000_gt.parquet","data/soccertrack_v2/st2_132877_1st_f015000_gt.parquet"])
def hs(paths, exclude=set(), players_only=True):
    h=[]; ch=[]; n=0
    for p in paths:
        b = pickle.load(open(p,"rb"))
        for pid, q in b["people"].items():
            if players_only and q["role"]!="player": continue
            if pid in exclude: continue
            if len(q["crops"])<4: continue
            h += q["h"]; ch += [c.shape[0] for c in q["crops"]]
    h=np.array(h); ch=np.array(ch)
    return len(h), h.min(), np.percentile(h,[1,5,50,95,99]).round(1), h.max(), "crop px:", ch.min(), np.percentile(ch,[1,5,50,95,99]).round(1), ch.max()
print("train", hs(sorted(glob.glob(".cache/soccertrack_v2/reid_train/*.pkl")), ex))
for p in [".cache/soccertrack_v2/reid_val/st2_118575_1st_f015000.pkl",".cache/soccertrack_v2/reid_report/st2_117093_1st_f015000.pkl",".cache/soccertrack_v2/reid_report/st2_132877_1st_f015000.pkl"]:
    print(p.split("/")[-1], hs([p]))
print("----")
def pct(paths, exclude=set()):
    h=[]
    for p in paths:
        b = pickle.load(open(p,"rb"))
        for pid, q in b["people"].items():
            if q["role"]!="player" or pid in exclude or len(q["crops"])<4: continue
            h += [c.shape[0] for c in q["crops"]]
    h=np.array(h)
    return {k: round(float(np.percentile(h,k)),1) for k in (2.5,5,10,25,50,75,90,95,97.5)}, "frac in 25-85:", round(float(((h>=25)&(h<=85)).mean()),3), "frac in 35-70:", round(float(((h>=35)&(h<=70)).mean()),3)
print("train", pct(sorted(glob.glob(".cache/soccertrack_v2/reid_train/*.pkl")), ex))
print("M3/M2/M10", pct([".cache/soccertrack_v2/reid_val/st2_118575_1st_f015000.pkl",".cache/soccertrack_v2/reid_report/st2_117093_1st_f015000.pkl",".cache/soccertrack_v2/reid_report/st2_132877_1st_f015000.pkl"]))
