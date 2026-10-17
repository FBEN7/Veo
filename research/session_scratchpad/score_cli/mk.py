"""Writes the synthetic sitting of score_ball_track._sitting to files (no real data)."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/user/Veo")
import score_ball_track as sb
d = Path(sys.argv[1])
key, judged, passes, scores = sb._sitting()
(d / "S").mkdir(exist_ok=True); (d / "S_key").mkdir(exist_ok=True)
(d / "S" / "judged.txt").write_text("".join(f"{n} {judged[n]}\n" for n in sorted(judged)))
(d / "S_key" / "key.json").write_text(json.dumps(key))
(d / "S_key" / "key_passes.json").write_text(json.dumps(passes))
(d / "S.json").write_text(json.dumps(scores))
