# Data in this repository

Committed on request on 2026-10-13, stored with Git LFS (`.gitattributes`). A
clone fetches the LFS objects only when needed: `git lfs pull --include "<path>"`.

| path | what it is | source and licence |
|---|---|---|
| `data/soccertrack_v2/` | 3-minute windows cut from SoccerTrack v2 matches (video, ground truth, actions) | SoccerTrack v2, Atom Scott et al., Hugging Face `atomscott/soccertrack-v2`, **CC BY 4.0** |
| `.cache/` | intermediate results built from those matches: ball strips and patches, labels, dense detections, re-ID crops, calibrations, model checkpoints trained on them | derived from SoccerTrack v2, **CC BY 4.0** (attribution above) |
| `output_st2_*/` | pipeline runs on SoccerTrack v2 windows | derived from SoccerTrack v2, **CC BY 4.0** |
| `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt` | stock Ultralytics COCO weights (unchanged) | Ultralytics, **AGPL-3.0** |
| `research/session_scratchpad/` | working files of the analysis sessions: protocols, logs, scripts, blind tile sheets with their judgements and keys, intermediate results | derived from SoccerTrack v2, **CC BY 4.0** |

Nothing from SoccerNet or other footage under a non-commercial or
no-redistribution licence is included; the `.gitignore` rules that keep it
out still apply.

The matches the protocols call test matches (SoccerTrack v2 `128057`, M7, and
`132831`, M9) were never opened and are not here.
