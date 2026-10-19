import sys, time
sys.path.insert(0, "/home/user/Veo")
from pathlib import Path
import pandas as pd
import eval_soccertrack_v2 as E
t0 = time.time()
clip = Path("/home/user/Veo/data/soccertrack_v2/st2_117093_1st_f015000.mp4")
out = Path("/home/user/Veo/output_st2_117093_1st_f015000/outline_only")
gt = E.truth(clip.with_name(clip.stem + "_gt.parquet"))
stride = 2
gt = gt[gt.frame % stride == 0]
side = gt.groupby("person").side.agg(lambda s: s.mode().iloc[0])
outfield = gt.groupby("person").role.agg(lambda s: s.mode().iloc[0]) == "player"
os_ = side[outfield]
det = E.boxes(pd.read_parquet(out / "tracks_merged.parquet"))
det = det[det.frame % stride == 0]
for how, m in (("feet", E.match_feet), ):
    person = m(det, gt)
    print(how, E.team_scores(det.team, person.map(os_), int((gt.role == "player").sum())), round(time.time()-t0,1))
