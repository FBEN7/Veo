"""Label regions for the weakly labelled ball patches
(`build_ball_patches.py`), and ball-free patches to go with them, for
`train_ball_heatmap.py`.

Per patch, a code per 4 x 4 cell: 1 the ball is somewhere here (a
positive bag), 0 ignore, -1 not the ball, -3 not the ball and easily
taken for it (other players, the actor's head). Regions are sized in
metres through the match's calibration -- 60 px is 3 m on the far side
and 0.7 m on the near one.

- At an action (PASS, DRIVE, HIGH PASS, CROSS, FREE KICK, TACKLE only:
  at an OUT, GOAL, THROW IN, HEADER, SHOT or BLOCK the ball is not at the
  feet): the union of the regions, 1.5 m on the ground and up to 0.5 m
  high, around the ball track's point and the actor's position; an ignore
  band of 0.5 m around them; everything else negative. Dropped if the
  actor is missing or more than 3 m from the track point.
- Between actions (gaps opened by a PASS or DRIVE, ball in play): a 2.5 m
  region, positive only -- the interpolated point is loose and the ball
  may be in the air, so nothing there is called negative. A DRIVE taken
  over by the same player is centred on the dribbler instead (1.5 m).
- Ball-free patches from the first-half training windows: at PASS and
  DRIVE frames, 20 m or more from the ball, half on players' feet, a fifth
  on pitch markings, the rest anywhere on the pitch; all negative.

A bag is kept only if three quarters of its region is inside the patch.

    python build_ball_labels.py --patches DIR --windows W1 W2 ...
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src import soccertrack_v2 as st
from src.ball_heatmap import STRIDE, ellipse, local_scale, pitch_to_image
from src.paths import DATA_DIR

SIZE = 256
CELLS = SIZE // STRIDE
KEEP_ACTION = {"PASS", "DRIVE", "HIGH PASS", "CROSS", "FREE KICK",
               "PLAYER SUCCESSFUL TACKLE"}
R_ACTION, R_BETWEEN, R_DRIBBLE, HEIGHT = 1.5, 2.5, 1.5, 0.5
MIN_INSIDE = 0.75
LANDMARKS = [(0, 0), (-41.5, 0), (41.5, 0), (-52.5, 0), (52.5, 0),
             (-52.5, -34), (-52.5, 34), (52.5, -34), (52.5, 34), (0, -34),
             (0, 34), (-36, 0), (36, 0)]


def region(cal, xy, x0, y0, radius, height=HEIGHT, grow=0.0):
    """Cells of an extended grid (patch +-1 patch) inside the image of a
    ground disk of `radius` metres around `xy`, up to `height` high, grown
    by `grow` metres; (mask over the extended grid, its patch slice)."""
    sx, sy = local_scale(xy, cal)
    uv = pitch_to_image([xy], cal)[0] - (x0 - SIZE, y0 - SIZE)
    r = radius + grow
    half_w = min(r * sx + 8, 112 + grow * sx)
    below = min(r * sy + 8, 112 + grow * sx)
    above = min(r * sy + height * sx + 8, 112 + grow * sx)
    return ellipse((3 * CELLS, 3 * CELLS), uv, half_w, below, above)


def inner(mask):
    return mask[CELLS:2 * CELLS, CELLS:2 * CELLS]


def body_box(codes, cal, xy, x0, y0, head_only=False):
    """Mark a player's body (or head band) at pitch `xy` as hard negative
    where the cells are plain negatives."""
    sx, sy = local_scale(xy, cal)
    u, v = pitch_to_image([xy], cal)[0] - (x0, y0)
    if head_only:
        hw, top, bot = 0.3 * sx, v - 1.9 * sx, v - 1.2 * sx
    else:
        hw, top, bot = 0.3 * sx + 10, v - 1.9 * sx, v + 0.3 * sy + 10
    c0, c1 = int(max((u - hw) // STRIDE, 0)), int(min((u + hw) // STRIDE + 1, CELLS))
    r0, r1 = int(max(top // STRIDE, 0)), int(min(bot // STRIDE + 1, CELLS))
    if c0 < c1 and r0 < r1:
        block = codes[r0:r1, c0:c1]
        block[block == -1] = -3


def half_context(match: str, half: int):
    ev = st.events(match)
    ev = ev[ev.half == half].sort_values("frame").reset_index(drop=True)
    z = np.load(io.BytesIO(st.open_stream(
        f"ball/{match}_{st._half(half)}_ball.npz").read()), allow_pickle=False)
    ball = pd.DataFrame({"x": z["x"], "y": z["y"], "status": z["status"]},
                        index=z["frame"].astype(int) - 1)
    gsr = st.gsr_table(match, half)
    gsr = gsr[gsr.role.isin(("player", "goalkeeper"))]
    return ev, ball, {f: g for f, g in gsr.groupby("frame")}, \
        st.calibration(match)


def label_patch(r, ev, ball, gsr, cal):
    """(codes 64x64 int8, positive_only) or None to drop."""
    f = int(r.frame)
    if f not in ball.index or f not in gsr:
        return None
    track = ball.loc[f, ["x", "y"]].to_numpy(float)
    people = gsr[f]
    if r.kind == "action":
        here = ev[ev.frame == f]
        here = here[here.label.isin(KEEP_ACTION)]
        if here.empty or here.player_id.isna().all():
            return None
        a = here.iloc[0]
        actor = people[people.player_id == a.player_id]
        if actor.empty:
            return None
        feet = actor[["x", "y"]].to_numpy(float)[0]
        if np.hypot(*(feet - track)) > 3.0:
            return None
        reg = region(cal, track, r.x0, r.y0, R_ACTION) | \
            region(cal, feet, r.x0, r.y0, R_ACTION)
        band = region(cal, track, r.x0, r.y0, R_ACTION, grow=0.5) | \
            region(cal, feet, r.x0, r.y0, R_ACTION, grow=0.5)
        if inner(reg).sum() < MIN_INSIDE * reg.sum():
            return None
        codes = np.where(inner(band), 0, -1).astype(np.int8)
        codes[inner(reg)] = 1
        for _, p in people.iterrows():
            xy = np.array([p.x, p.y], float)
            if p.player_id == a.player_id:
                body_box(codes, cal, xy, r.x0, r.y0, head_only=True)
            elif np.hypot(*(xy - feet)) > R_ACTION + 1.0:
                body_box(codes, cal, xy, r.x0, r.y0)
        codes[inner(reg)] = 1
        return codes, False
    # Between actions: the gap's opener, the ball in play.
    before = ev[ev.frame <= f]
    after = ev[ev.frame > f]
    if before.empty or before.iloc[-1].label not in ("PASS", "DRIVE"):
        return None
    if int(ball.loc[f, "status"]) not in (1, 2):
        return None
    centre, radius = track, R_BETWEEN
    opener = before.iloc[-1]
    if (opener.label == "DRIVE" and not after.empty
            and after.iloc[0].player_id == opener.player_id):
        who = people[people.player_id == opener.player_id]
        if not who.empty:
            centre, radius = who[["x", "y"]].to_numpy(float)[0], R_DRIBBLE
    reg = region(cal, centre, r.x0, r.y0, radius)
    if inner(reg).sum() < MIN_INSIDE * reg.sum():
        return None
    codes = np.zeros((CELLS, CELLS), np.int8)
    codes[inner(reg)] = 1
    return codes, True


def negatives(window: str, out: Path, per_frame: int = 4, seed: int = 0):
    """Ball-free patches from a first-half training window."""
    rng = np.random.default_rng(seed)
    info = __import__("json").loads(
        (DATA_DIR / "soccertrack_v2" / f"{window}.json").read_text())
    match, half, start = info["match"], info["half"], info["start_frame"]
    cal = st.calibration(match)
    z = np.load(io.BytesIO(st.open_stream(
        f"ball/{match}_{st._half(half)}_ball.npz").read()), allow_pickle=False)
    ball = pd.DataFrame({"x": z["x"], "y": z["y"]},
                        index=z["frame"].astype(int) - 1 - start)
    lab = pd.read_csv(DATA_DIR / "soccertrack_v2" / f"{window}_events.csv")
    frames = sorted(set(lab[lab.label.isin(("PASS", "DRIVE"))].frame))
    gt = pd.read_parquet(DATA_DIR / "soccertrack_v2" / f"{window}_gt.parquet")
    gt = {f: g for f, g in gt.groupby("frame")}
    poly = np.asarray(info["pitch"], np.float32)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    cap = cv2.VideoCapture(str(DATA_DIR / "soccertrack_v2" / f"{window}.mp4"))
    rows, codes, idx = [], [], 0
    want = set(frames)
    while frames and idx <= max(frames):
        ok, image = cap.read()
        if not ok:
            break
        if idx in want and idx in ball.index and idx in gt:
            anchor = ball.loc[idx, ["x", "y"]].to_numpy(float)
            au = pitch_to_image([anchor], cal)[0][0]
            people = gt[idx][["x", "y"]].to_numpy(float)
            far = people[np.hypot(*(people - anchor).T) >= 20]
            picks = []
            for k in range(per_frame):
                choice = rng.random()
                if choice < 0.5 and len(far):
                    xy = far[rng.integers(len(far))]
                elif choice < 0.7:
                    xy = np.array(LANDMARKS[rng.integers(len(LANDMARKS))], float)
                else:
                    uv = rng.uniform(lo, hi)
                    xy = st.image_to_pitch([uv], cal)[0]
                if not np.isfinite(xy).all() or np.hypot(*(xy - anchor)) < 20:
                    continue
                uv = pitch_to_image([xy], cal)[0]
                if abs(uv[0] - au) <= 150:
                    continue
                picks.append(uv + rng.integers(-64, 65, 2))
            H, W = image.shape[:2]
            for uv in picks:
                cx = int(np.clip(uv[0], SIZE // 2, W - SIZE // 2))
                cy = int(np.clip(uv[1], SIZE // 2, H - SIZE // 2))
                x0, y0 = cx - SIZE // 2, cy - SIZE // 2
                name = f"neg_{window}_{idx:05d}_{x0}_{y0}.jpg"
                cv2.imwrite(str(out / name), image[y0:y0 + SIZE, x0:x0 + SIZE],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                c = np.full((CELLS, CELLS), -1, np.int8)
                for xy in people:
                    body_box(c, cal, xy, x0, y0)
                rows.append({"file": name, "match": match, "half": half,
                             "frame": idx + start, "kind": "negative",
                             "x0": x0, "y0": y0, "positive_only": False})
                codes.append(c)
        idx += 1
    cap.release()
    return rows, codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True)
    ap.add_argument("--windows", nargs="*", default=[],
                    help="first-half training windows for ball-free patches")
    args = ap.parse_args()
    root = Path(args.patches)
    table = pd.read_parquet(root / "patches.parquet")
    table["match"] = table.match.astype(str)
    rows, codes = [], []
    for (match, half), part in table.groupby(["match", "half"]):
        ctx = half_context(match, int(half))
        kept = 0
        for r in part.itertuples():
            out = label_patch(r, *ctx)
            if out is None:
                continue
            c, pos_only = out
            rows.append({"file": r.file, "match": match, "half": int(half),
                         "frame": int(r.frame), "kind": r.kind, "x0": r.x0,
                         "y0": r.y0, "positive_only": pos_only})
            codes.append(c)
            kept += 1
        print(f"  {match} half {half}: {kept} of {len(part)} patches labelled",
              flush=True)
    for w in args.windows:
        r, c = negatives(w, root)
        rows += r
        codes += c
        print(f"  {w}: {len(r)} ball-free patches", flush=True)
    pd.DataFrame(rows).to_parquet(root / "labels.parquet")
    np.save(root / "labels_codes.npy", np.stack(codes))
    print(f"  {len(rows)} patches labelled", flush=True)


if __name__ == "__main__":
    main()
