import pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/user/Veo")
import identify_players as ip
from src import player_identity as pid

d = Path(sys.argv[1]); d.mkdir(parents=True, exist_ok=True)
for f in d.iterdir(): f.unlink()
lookA = np.array([1.0, 0.0]); 
t = pid.Track(track_id=1, team="A", frames=np.arange(3))
t.visible = np.ones(3); t.look = lookA
(d / "identity_tracks.pkl").write_bytes(pickle.dumps({1: t}))
# No clip.json / tracks_merged.parquet: any recompute would raise.
class EmbedderB: pass
out = ip.read_tracks(d, model=None, embedder=EmbedderB())
print("returned look (A's):", out[1].look, "-- no recompute with embedder B")
