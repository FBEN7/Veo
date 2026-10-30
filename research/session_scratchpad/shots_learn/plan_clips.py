"""The clips of shots_learn/protocol.json, planned here, cut on Kaggle.

For each match half of a set, the -15 s / +10 s spans around its BAS SHOT
and GOAL, merged (kaggle/cut_shot_clips.spans), cut with MARGIN frames each
side. Here, without the video: <name>.json (as fetch_soccertrack_v2.fetch
writes it, with the pitch outline; frames, width and height are set by the
Kaggle cutter from what it writes), <name>_gt.parquet (GSR of the cut's
frames) and <name>_events.csv (BAS actions of the cut, with actors), into
OUT/soccertrack_v2; the plan to clips_<set>.json here. The mp4 is cut on
Kaggle (kaggle/cut_on_kaggle.py, the same st.cut_clip).

    python plan_clips.py train
    python plan_clips.py test     # only once frozen.json exists
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/kaggle"))
from cut_shot_clips import LABELS, MARGIN, spans  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402

HERE = Path(__file__).resolve().parent
TRAIN = ["117092", "117093", "118575", "118576", "118578", "128058", "132877"]
TEST = ["118577"]
SIZE = {m: (4096, 1080) for m in TRAIN + TEST}


def pitch_of(match: str, half: int, w: int, h: int) -> list:
    """From a processed window of the same half when there is one (the view
    is fixed), else from the release's pitch lines."""
    for p in sorted((ROOT / "data/soccertrack_v2").glob(f"st2_{match}_{st._half(half)}_f*.json")):
        info = json.loads(p.read_text())
        if info.get("pitch") and (info["width"], info["height"]) == (w, h):
            return info["pitch"]
    return st.pitch_outline(match, half, w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=("train", "test"))
    ap.add_argument("--out", default=str(ROOT / "data/shot_clips"))
    args = ap.parse_args()
    if args.set == "test" and not (HERE / "frozen.json").exists():
        raise SystemExit("the M5 clips are planned only after frozen.json exists")
    out = Path(args.out) / "soccertrack_v2"
    out.mkdir(parents=True, exist_ok=True)
    plan = {"set": args.set, "margin_frames": MARGIN, "clips": []}
    for match in TRAIN if args.set == "train" else TEST:
        ev = st.events(match)
        for half in (1, 2):
            att = ev[(ev.half == half) & ev.label.isin(LABELS)]
            if att.empty:
                continue
            gsr = st.gsr_table(match, half)
            w, h = SIZE[match]
            pitch = pitch_of(match, half, w, h)
            for a, b, fs in spans(att.frame.astype(int).tolist()):
                start = max(a - MARGIN, 0)
                count = b + MARGIN + 1 - start
                name = f"st2_{match}_{st._half(half)}_f{start:06d}"
                g = gsr[(gsr.frame >= start) & (gsr.frame < start + count)].copy()
                g["frame"] -= start
                g.to_parquet(out / f"{name}_gt.parquet")
                e = ev[(ev.half == half) & (ev.frame >= start) & (ev.frame < start + count)]
                e = st.with_actors(e, gsr)
                e["frame"] -= start
                e.to_csv(out / f"{name}_events.csv", index=False)
                info = {"match": match, "half": half, "start_frame": start, "frames": count,
                        "fps": st.FPS, "width": w, "height": h, "pitch": pitch, "name": name,
                        "dataset": f"SoccerTrack v2 {st.REVISION}", "licence": "CC BY 4.0",
                        "paper_label": st.MATCHES[match][0], "split": st.MATCHES[match][1],
                        "gt": f"{name}_gt.parquet", "events": f"{name}_events.csv",
                        "cut": {"start": start, "count": count, "crf": 18}}
                (out / f"{name}.json").write_text(json.dumps(info, indent=1))
                plan["clips"].append({"name": name, "match": match, "half": half,
                                      "cut_start": start, "cut_frames": count,
                                      "span_half": [a, b], "span_local": [a - start, b - start],
                                      "attempts_half": fs})
            print(f"{match} half {half}: {len(att)} attempts", flush=True)
    plan["n_clips"] = len(plan["clips"])
    plan["n_attempts"] = sum(len(c["attempts_half"]) for c in plan["clips"])
    (HERE / f"clips_{args.set}.json").write_text(json.dumps(plan, indent=1))
    print(plan["n_clips"], "clips,", plan["n_attempts"], "attempts")


if __name__ == "__main__":
    main()
