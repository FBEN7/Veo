import pandas as pd, glob, pickle, os
root = "/home/user/Veo/data/soccertrack_v2"
for p in sorted(glob.glob(f"{root}/*_gt.parquet")):
    g = pd.read_parquet(p, columns=["track_id","player_id"])
    print(os.path.basename(p), "rows", len(g), "missing", int(g.player_id.isna().sum()))
pk = sorted(glob.glob("/home/user/Veo/.cache/**/*.pkl", recursive=True))
print(len(pk), "pkl files")
bad = 0
for f in pk:
    try:
        b = pickle.loads(open(f,"rb").read())
    except Exception as e:
        continue
    if isinstance(b, dict) and "people" in b:
        keys = list(b["people"])
        odd = [k for k in keys if not str(k).isdigit()]
        print(f, len(keys), "odd keys:", odd)
