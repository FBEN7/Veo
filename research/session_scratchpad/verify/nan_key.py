import pandas as pd, glob
root = "/home/user/Veo/data/soccertrack_v2"
p = sorted(glob.glob(f"{root}/st2_118575*_gt.parquet"))[0]
raw = pd.read_parquet(p)
print(p, raw.dtypes.to_dict())
print("missing player_id:", raw.player_id.isna().sum())
tids = sorted(raw.track_id.unique())[:5]
raw.loc[raw.track_id.isin([tids[1], tids[3]]), "player_id"] = None
who = raw.groupby("track_id").agg(
    player_id=("player_id", lambda s: s.dropna().mode().iloc[0]
               if s.notna().any() else None),
    role=("role", lambda s: s.mode().iloc[0]),
    side=("side", lambda s: s.mode().iloc[0]))
print(who.dtypes.to_dict())
for t in [tids[0], tids[1], tids[3]]:
    w = who.loc[t]
    key = str(w.player_id) if w.player_id is not None else f"t{t}"
    print(t, repr(w.player_id), w.player_id is not None, key)
# what if all values none
raw2 = raw.copy(); raw2["player_id"] = None
who2 = raw2.groupby("track_id").agg(player_id=("player_id", lambda s: s.dropna().mode().iloc[0] if s.notna().any() else None))
print(who2.dtypes.to_dict(), repr(who2.player_id.iloc[0]))
