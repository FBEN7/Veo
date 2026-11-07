"""Cut the shot clips of protocol_shots.json for the Kaggle GPU runs.

Clips (protocol): -15 s / +10 s around each BAS SHOT and GOAL of a half,
merged where they overlap. The evaluated span of a merged clip is
[first L - 375, last L + 250] (half frames, both ends included). The cut
adds MARGIN frames on each side (owner decision, 2026-10-20): v3's motion
input takes the background of every detected frame from the frames that are
multiples of 50 within 350 of it (src/motion_bg.py), and the pipeline's
tracker settles before the span. Shots are evaluated only inside the span.

Sets:
  dev   M7 1st (128057 half 1) and M9 2nd (132831 half 2), the 26 gate
        attempts; sealed matches, opened here through src/unseal.py (each
        opening logged there); VEO_UNSEAL never leaves this machine.
  test  M1 (117092) both halves, the 29 attempts: only after the dev choice
        is frozen in frozen_shots.json (protocol "order"); refused before.

Each clip is cut with fetch_soccertrack_v2.fetch (mp4, _gt.parquet,
_events.csv, .json with the pitch outline) into OUT/soccertrack_v2, named
st2_<match>_<half>_f<first cut frame>; a name that is a processed window's
is refused. The plan (clips, spans, attempts) goes to
shot_clips_<set>.json here; already cut clips are skipped.

    VEO_UNSEAL=... python cut_shot_clips.py dev [--out data/shot_clips]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from fetch_soccertrack_v2 import fetch  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.unseal import unsealed  # noqa: E402

HERE = Path(__file__).resolve().parent
SG = ROOT / "research/session_scratchpad/shots_goals"
BEFORE, AFTER = 15 * st.FPS, 10 * st.FPS
MARGIN = 400                                     # 350 + 50: the background reach
SETS = {"dev": [("128057", 1), ("132831", 2)], "test": [("117092", 1), ("117092", 2)]}
SEALED = {"128057", "132831"}
LABELS = ("SHOT", "GOAL")


def spans(frames: list[int]) -> list[list[int]]:
    out = []
    for f in sorted(frames):
        a, b = f - BEFORE, f + AFTER
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
            out[-1][2].append(f)
        else:
            out.append([a, b, [f]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set", choices=sorted(SETS))
    ap.add_argument("--out", default=str(ROOT / "data/shot_clips"))
    args = ap.parse_args()
    if args.set == "test" and not (SG / "frozen_shots.json").exists():
        raise SystemExit("test clips are cut only after frozen_shots.json exists")
    out = Path(args.out)
    windows = {p.stem for p in (ROOT / "data/soccertrack_v2").glob("st2_*.json")}
    plan = {"set": args.set, "before_s": 15, "after_s": 10, "margin_frames": MARGIN,
            "clips": []}
    for match, half in SETS[args.set]:
        if match in SEALED and not unsealed(match, f"shots {args.set}: cut clips for Kaggle"):
            raise SystemExit(f"{match} is sealed: set VEO_UNSEAL (src/unseal.py)")
        ev = st.events(match)
        att = ev[(ev.half == half) & ev.label.isin(LABELS)]
        for a, b, fs in spans(att.frame.astype(int).tolist()):
            start = max(a - MARGIN, 0)
            stop = b + MARGIN + 1                        # cut_clip stops at the half's end
            name = f"st2_{match}_{st._half(half)}_f{start:06d}"
            if name in windows:
                raise SystemExit(f"{name} is a processed window's name")
            clip = {"name": name, "match": match, "half": half, "cut_start": start,
                    "cut_frames": stop - start, "span_half": [a, b],
                    "span_local": [a - start, b - start], "attempts_half": fs}
            if not (out / "soccertrack_v2" / f"{name}.json").exists():
                fetch(match, half, start / (60 * st.FPS), (stop - start) / (60 * st.FPS), out / "soccertrack_v2")
            info = json.loads((out / "soccertrack_v2" / f"{name}.json").read_text())
            clip["frames_written"] = int(info["frames"])
            plan["clips"].append(clip)
            print(f"{name}: {len(fs)} attempts, {clip['frames_written']} frames", flush=True)
    plan["n_attempts"] = sum(len(c["attempts_half"]) for c in plan["clips"])
    (HERE / f"shot_clips_{args.set}.json").write_text(json.dumps(plan, indent=1))
    print(f"{len(plan['clips'])} clips, {plan['n_attempts']} attempts")


if __name__ == "__main__":
    main()
