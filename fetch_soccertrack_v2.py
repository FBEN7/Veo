"""Cut SoccerTrack v2 windows with their ground truth (`src/soccertrack_v2.py`).

For a match, a half and a window, writes to `<data>/soccertrack_v2/`:

- `<name>.mp4`: the window of the fixed panoramic video, full resolution,
  H.264 (plays in a browser, so the labelling pages can use it);
- `<name>_gt.parquet`: every person on every frame of the window -- frame
  (from the window's start), track_id, player_id, role, jersey, side, pitch
  x / y in metres, image box bx, by, bw, bh;
- `<name>_events.csv`: the ball actions in the window -- frame, label, kind
  (this project's type), side, player_id, the actor's jersey and pitch
  position;
- `<name>.json`: where the window comes from.

The half's GSR is streamed once (~2.7 GB, a few minutes) and kept as
parquet in the cache; the video is read by range requests, only the
window's minutes. Needs HF_TOKEN (the dataset is gated). The files are
derived from the dataset (CC BY 4.0) and stay out of the repository.

    python fetch_soccertrack_v2.py --match 117093 --half 1 --start 10 --minutes 3
    python fetch_soccertrack_v2.py --list
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src import soccertrack_v2 as st
from src.paths import DATA_DIR


def sizes():
    """The size of each match's videos and GSR files, to plan disk."""
    import requests

    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {st.token()}"
    for match, (label, split) in st.MATCHES.items():
        out = []
        for half in (1, 2):
            for path in (st.video_path(match, half),
                         f"gsr/{match}/{match}_{st._half(half)}.json"):
                r = s.head(st.url(path), allow_redirects=True, timeout=60)
                n = int(r.headers.get("Content-Length", 0)) if r.ok else 0
                out.append(f"{path.split('/')[-1]} {n / 1e9:.2f} GB")
        print(f"  {match} ({label}, {split}): " + ", ".join(out), flush=True)


def fetch(match: str, half: int, start_min: float, minutes: float,
          root: Path) -> Path:
    start = int(round(start_min * 60 * st.FPS))
    count = int(round(minutes * 60 * st.FPS))
    name = f"st2_{match}_{st._half(half)}_f{start:06d}"
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    gsr = st.gsr_table(match, half)
    print(f"  GSR {match} half {half}: {len(gsr)} records, "
          f"{gsr.frame.nunique()} frames  [{time.time() - t0:.0f} s]",
          flush=True)
    print(f"  pitch y grows towards the camera: {st.y_towards_camera(gsr)}",
          flush=True)
    t0 = time.time()
    info = st.cut_clip(match, half, start, count, root / f"{name}.mp4")
    # The ground truth of the frames written: a window may run past the
    # half's end.
    n = info["frames"]
    win = gsr[(gsr.frame >= start) & (gsr.frame < start + n)].copy()
    win["frame"] -= start
    win.to_parquet(root / f"{name}_gt.parquet")
    ev = st.events(match)
    ev = ev[(ev.half == half) & (ev.frame >= start) & (ev.frame < start + n)]
    ev = st.with_actors(ev, gsr)
    ev["frame"] -= start
    ev.to_csv(root / f"{name}_events.csv", index=False)
    info.update(name=name, dataset=f"SoccerTrack v2 {st.REVISION}",
                licence="CC BY 4.0", paper_label=st.MATCHES[match][0],
                split=st.MATCHES[match][1], gt=f"{name}_gt.parquet",
                events=f"{name}_events.csv")
    (root / f"{name}.json").write_text(json.dumps(info, indent=1))
    print(f"  {name}: {n} frames {info['width']}x{info['height']}, "
          f"{win.track_id.nunique()} people, {len(ev)} actions "
          f"({ev.label.value_counts().to_dict()})  [{time.time() - t0:.0f} s]",
          flush=True)
    return root / f"{name}.mp4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", choices=sorted(st.MATCHES))
    ap.add_argument("--half", type=int, choices=(1, 2), default=1)
    ap.add_argument("--start", type=float, default=0.0,
                    help="minutes into the half")
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--out", default=str(DATA_DIR / "soccertrack_v2"))
    ap.add_argument("--list", action="store_true",
                    help="print each match's file sizes and stop")
    args = ap.parse_args()
    if args.list:
        sizes()
        return
    if not args.match:
        ap.error("--match is needed (or --list)")
    fetch(args.match, args.half, args.start, args.minutes, Path(args.out))


if __name__ == "__main__":
    main()
