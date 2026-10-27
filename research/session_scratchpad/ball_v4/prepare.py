"""The clicked balls as training strips and test frames (ball detector v4).

Input: the click page's db rows (exported by ArtifactData list into
clicks/<doc>.json) and ball_clicks/manifest.json. Per clicked frame the
ball's full-frame position is the click plus the crop's offset.

- Folds by match, M6 (118578) never trained on (it holds the guard
  windows): fold A trains on M1, M2, M3, M4 (117092, 117093, 118575,
  118577) and is tested on M5, M8, M10 (118576, 128058, 132877) and M6;
  fold B the other way round.
- Strips (cut on Kaggle, build_ball_strips.cut over each clip): per
  "ball" frame 3 crops placed around the ball (kind "verified", codes 1
  within 12 px of the ball on both axes, 0 to 20 px, -1 elsewhere: the
  frame's ball is known) and one crop away from it (kind "verified_neg",
  all -1, the ball's 40 px box outside it). "none" and "unsure" frames
  give no strip.
- Test frames: every clicked frame (ball / none / unsure) with its truth,
  and the verified balls of the M6 window st2_118578_2nd_f015000 (fold
  "guard_w"; v3 was selected on M6, so this guard favours v3b slightly).

Writes stage/targets.parquet (one row per strip, with clip and fold),
stage/codes.npy (aligned) and stage/test_frames.parquet.

    python prepare.py CLICKS_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from build_ball_strips import place  # noqa: E402
from src.ball_heatmap import pitch_box  # noqa: E402

CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
FOLDS = {"A": {"117092", "117093", "118575", "118577"}, "B": {"118576", "128058", "132877"}}
GUARD = "118578"
GUARD_WINDOW = "st2_118578_2nd_f015000"        # v3's verified balls (ball_verified/A)
SIZE, STRIDE = 320, 4
CELLS = SIZE // STRIDE
BALL_PX, RING_PX = 12.0, 20.0
PLACEMENTS = 3
NEG_CLEAR = 40.0
SEED = 0


def fold_of(match: str) -> str:
    for k, ms in FOLDS.items():
        if match in ms:
            return k
    return "guard" if match == GUARD else "none"


def codes_for(x0: int, y0: int, bx: float | None, by: float | None) -> np.ndarray:
    c = np.full((CELLS, CELLS), -1, np.int8)
    if bx is None:
        return c
    cx = x0 + STRIDE * np.arange(CELLS) + STRIDE / 2
    cy = y0 + STRIDE * np.arange(CELLS) + STRIDE / 2
    dx, dy = np.abs(cx[None, :] - bx), np.abs(cy[:, None] - by)
    c[(dx <= RING_PX) & (dy <= RING_PX)] = 0
    c[(dx <= BALL_PX) & (dy <= BALL_PX)] = 1
    return c


def main():
    clicks_dir = Path(sys.argv[1])
    man = {s["id"]: s for s in json.loads((ROOT / "research/session_scratchpad/ball_clicks/manifest.json").read_text())}
    docs = [json.loads(p.read_text()) for p in sorted(clicks_dir.rglob("*.json"))]
    docs = [d.get("data", d) for d in docs]
    rng = np.random.default_rng(SEED)
    targets, codes, tests = [], [], []
    for d in docs:
        s = man[d["shot"]]
        info = json.loads((CLIPS / f"{s['unit']}.json").read_text())
        start = int(info["start_frame"])
        local = int(d["frame"])
        bx = by = None
        if d["status"] == "ball" and d.get("x") is not None:
            bx, by = s["x0"] + float(d["x"]), s["y0"] + float(d["y"])
        tests.append({"unit": s["unit"], "match": s["match"], "fold": fold_of(s["match"]),
                      "shot": d["shot"], "k": int(d["k"]), "frame": local, "status": d["status"],
                      "u": bx, "v": by, "spotter_hit": s["spotter_hit"]})
        if bx is None or fold_of(s["match"]) not in FOLDS:
            continue
        box = pitch_box((int(info["height"]), int(info["width"])), info["pitch"])
        base = {"unit": s["unit"], "match": s["match"], "half": int(info["half"]),
                "fold": fold_of(s["match"]), "frame": start + local, "u": bx, "v": by}
        for _ in range(PLACEMENTS):
            x0, y0 = place((bx - BALL_PX, by - BALL_PX, bx + BALL_PX, by + BALL_PX), rng, box)
            targets.append(dict(base, kind="verified", x0=int(x0), y0=int(y0)))
            codes.append(codes_for(x0, y0, bx, by))
        for _ in range(20):                       # one crop clear of the ball's 40 px box
            x0 = int(rng.integers(box[0], box[2] - SIZE))
            y0 = int(rng.integers(box[1], max(box[1] + 1, box[3] - SIZE)))
            if not (x0 - NEG_CLEAR <= bx <= x0 + SIZE + NEG_CLEAR and y0 - NEG_CLEAR <= by <= y0 + SIZE + NEG_CLEAR):
                targets.append(dict(base, kind="verified_neg", x0=x0, y0=y0))
                codes.append(codes_for(x0, y0, None, None))
                break
    g = pd.read_parquet(ROOT / ".cache/soccertrack_v2/ball_verified/A/frames.parquet")
    g = g[(g.window == GUARD_WINDOW) & (g.status == "ball")]
    tests += [{"unit": GUARD_WINDOW, "match": GUARD, "fold": "guard_w", "shot": "", "k": -1,
               "frame": int(r.frame), "status": "ball", "u": float(r.u), "v": float(r.v),
               "spotter_hit": True} for r in g.itertuples()]
    st = HERE / "stage"
    st.mkdir(exist_ok=True)
    t = pd.DataFrame(targets)
    t.to_parquet(st / "targets.parquet")
    np.save(st / "codes.npy", np.stack(codes) if codes else np.zeros((0, CELLS, CELLS), np.int8))
    tf = pd.DataFrame(tests)
    tf.to_parquet(st / "test_frames.parquet")
    print("clicked frames", len(tf), tf.status.value_counts().to_dict())
    print("strips", len(t), t.groupby(["fold", "kind"]).size().to_dict() if len(t) else {})


if __name__ == "__main__":
    main()
