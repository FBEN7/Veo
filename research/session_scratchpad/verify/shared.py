import glob, pandas as pd, itertools
M = {"117092":"M1","117093":"M2","118575":"M3","118576":"M4","118577":"M5","118578":"M6","128058":"M8","132877":"M10"}
ids = {}; ids1 = {}
for f in glob.glob(".cache/soccertrack_v2/gsr/*.parquet")+glob.glob("data/soccertrack_v2/*_gt.parquet"):
    name = f.split("/")[-1]
    m = name.replace("st2_","").split("_")[0]
    half = name.replace("st2_","").split("_")[1].split(".")[0]
    g = pd.read_parquet(f)
    s = set(g.player_id.dropna().astype(str))
    ids.setdefault(m, set()).update(s)
    if half == "1st":
        ids1.setdefault(m, set()).update(s)
    print(name, len(s), "roles:", g.role.value_counts().to_dict() if "role" in g else "")
for a,b in itertools.combinations(sorted(ids, key=lambda k: int(M[k][1:])),2):
    n = len(ids[a]&ids[b])
    if n: print(M[a], M[b], "shared (all halves)", n, " 1st halves", len(ids1[a]&ids1[b]))
excl_files = [".cache/soccertrack_v2/gsr/117093_1st.parquet",".cache/soccertrack_v2/gsr/118575_1st.parquet",".cache/soccertrack_v2/gsr/132877_1st.parquet","data/soccertrack_v2/st2_117093_1st_f015000_gt.parquet","data/soccertrack_v2/st2_118575_1st_f015000_gt.parquet","data/soccertrack_v2/st2_132877_1st_f015000_gt.parquet"]
ex=set()
for f in excl_files: ex |= set(pd.read_parquet(f, columns=["player_id"]).player_id.dropna().astype(str))
print("excluded", len(ex))
for m in ids: print(M[m], "ids", len(ids[m]), "in excluded", len(ids[m]&ex))
