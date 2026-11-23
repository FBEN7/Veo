"""identity step 2a (plan.json, hashed first): part 1 (box heights, no video)
and the crop sheets of parts 2 and 3. Writes part1.json, sheets/*.png and
key.json (the numbers: not looked at before readings.json is committed).

    VEO_UNSEAL=... python sheets.py
"""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/player_stitch"))
import stitch as T  # noqa: E402

SCR = Path("/tmp/claude-0/-home-user-veo/b69e5b60-e0fe-5920-8f15-285768d7ea57/scratchpad")
SC = ROOT / "data/shot_clips/soccertrack_v2"
SOURCES = [("117093", ROOT / "data/soccertrack_v2/st2_117093_1st_f015000.mp4", ROOT / "data/soccertrack_v2/st2_117093_1st_f015000_gt.parquet"),
           ("118576", SCR / "shotvid/st2_118576_1st_f050779.mp4", SC / "st2_118576_1st_f050779_gt.parquet")]
SOURCES += [("132831", p, p.with_name(p.stem + "_gt.parquet")) for p in sorted(SC.glob("st2_132831_*.mp4"))]
TREE = {"st2_117093_1st_f015000": ROOT / ".cache/gpu_shots_windows"}  # addendum 1: our detections frame the crops
MATCH_M, ALONE_M = 2.0, 3.0
BANDS = [(0, 40, "<40"), (40, 60, "40-60"), (60, 80, "60-80"), (80, 1e9, ">=80")]
PER_BAND, STEP, SEG_N = 30, 50, 10
OUT = HERE / "sheets"


def check():
    out = {}
    for name in ("plan.json", "addendum1.json"):
        want = (HERE / f"{name}.sha256").read_text().split()[0]
        got = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"{name} does not match its sha256")
        out[name] = got
    return out


def part1():
    rows = []
    for name, match, _, _ in [u for u in T.CE.unit_list() if u[2] is None]:  # noqa: B007
        info = T.S.LS.sd.info_of(name)
        m = pd.read_parquet(tree_of(name) / name / f"output_{name}" / "metric_v3s_link.parquet")
        g = m[(m.cls == "player") & m.team.isin(["team_A", "team_B"])]
        rows.append(pd.DataFrame({"match": match, "bh": g.crop_h.to_numpy(), "width": info["width"]}))
    d = pd.concat(rows)
    per = {m: {"width": int(x.width.iloc[0]), "median": float(x.bh.median()), "p10": float(x.bh.quantile(0.1)),
               "p90": float(x.bh.quantile(0.9))} for m, x in d.groupby("match")}
    return {"per match": per, "all: median": float(d.bh.median()),
            "share of player-frames with bh >=": {str(t): round(float((d.bh >= t).mean()), 3) for t in (40, 60, 80)}}


def tree_of(name):
    for u in T.CE.unit_list():
        if u[0] == name:
            return u[3]
    return TREE[name]


def matched(gt, ours, frames):
    """addendum 1: truth outfield players matched one to one to our team-labelled detections."""
    from scipy.optimize import linear_sum_assignment
    out = []
    gt = gt.assign(xc=gt.x + 52.5, yc=gt.y + 34.0)
    G, O = dict(tuple(gt.groupby("frame"))), dict(tuple(ours.groupby("frame")))
    for f in frames:
        if f not in G or f not in O:
            continue
        g, o = G[f], O[f]
        d = np.hypot(g.xc.to_numpy()[:, None] - o.x.to_numpy()[None], g.yc.to_numpy()[:, None] - o.y.to_numpy()[None])
        r, c = linear_sum_assignment(d)
        for i, j in zip(r, c):
            if d[i, j] > MATCH_M or g.role.iloc[i] != "player":
                continue
            if np.sort(d[:, j])[1:2].min(initial=1e9) <= ALONE_M:
                continue
            q = o.iloc[j]
            out.append({"frame": f, "player_id": g.player_id.iloc[i], "jersey": g.jersey.iloc[i],
                        "px": q.px_raw, "py": q.py_raw, "bw": q.crop_w, "bh": q.crop_h})
    return pd.DataFrame(out)


def band(h):
    return next(b[2] for b in BANDS if b[0] <= h < b[1])


def pool():
    out = []
    for k, (match, vid, gtp) in enumerate(SOURCES):
        name = vid.stem
        m = pd.read_parquet(tree_of(name) / name / f"output_{name}" / "metric_v3s_link.parquet")
        ours = m[(m.cls == "player") & m.team.isin(["team_A", "team_B"]) & m.x.notna()]
        r0 = int(ours.frame.min()) % 2
        frames = [f for f in sorted(ours.frame.unique()) if f % STEP == r0]
        out.append(matched(pd.read_parquet(gtp), ours, frames).assign(match=match, src=k))
    p = pd.concat(out, ignore_index=True)
    p["band"] = p.bh.map(band)
    return p


def crop(img, r):
    pw, ph = 0.15 * r.bw, 0.15 * r.bh
    x0, y0 = int(max(r.px - r.bw / 2 - pw, 0)), int(max(r.py - r.bh - ph, 0))
    x1, y1 = int(min(r.px + r.bw / 2 + pw, img.shape[1])), int(min(r.py + ph, img.shape[0]))
    c = img[y0:y1, x0:x1]
    return cv2.resize(c, (c.shape[1] * 4, c.shape[0] * 4), interpolation=cv2.INTER_CUBIC)


def frames_needed(sel):
    need = {}
    for r in sel.itertuples():
        need.setdefault(r.src, set()).add(r.frame)
    imgs = {}
    for k, fs in need.items():
        cap = cv2.VideoCapture(str(SOURCES[k][1]))
        f, last = 0, max(fs)
        while f <= last:
            ok = cap.grab()
            if not ok:
                break
            if f in fs:
                imgs[(k, f)] = cap.retrieve()[1]
            f += 1
        cap.release()
    return imgs


def sheet(cells, cols, cw, ch, path):
    rows = (len(cells) + cols - 1) // cols
    canvas = np.full((rows * ch, cols * cw, 3), 255, np.uint8)
    for i, (label, c) in enumerate(cells):
        s = min(1.0, (cw - 4) / c.shape[1], (ch - 26) / c.shape[0])
        c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)), interpolation=cv2.INTER_AREA) if s < 1 else c
        y, x = (i // cols) * ch, (i % cols) * cw
        canvas[y + 24:y + 24 + c.shape[0], x + 2:x + 2 + c.shape[1]] = c
        cv2.putText(canvas, label, (x + 4, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imwrite(str(path), canvas)


def main():
    sha = check()
    from src.unseal import unsealed
    for m in T.S.V3.NEW_PLAYERS:
        if not unsealed(m, "identity step 2a: box heights of the M7 / M9 windows, 132831 crops (development use, seal lifted 2026-11-05)"):
            raise SystemExit("VEO_UNSEAL needed")
    (HERE / "part1.json").write_text(json.dumps({"sha256": sha, **part1()}, indent=1))
    rng = np.random.default_rng(0)
    p = pool()
    # part 2: up to 30 per band, evenly over the matches
    picks = []
    for _, _, b in BANDS:
        pb = p[p.band == b].drop_duplicates(["src", "frame", "player_id"])
        per_m = {m: x.sample(frac=1, random_state=int(rng.integers(1 << 30))) for m, x in pb.groupby("match")}
        got, i = [], 0
        while len(got) < PER_BAND and any(i < len(x) for x in per_m.values()):
            for x in per_m.values():
                if i < len(x) and len(got) < PER_BAND:
                    got.append(x.iloc[i])
            i += 1
        picks += got
    s2 = pd.DataFrame(picks).sample(frac=1, random_state=0).reset_index(drop=True)
    s2["id"] = [f"A{i:03d}" for i in range(len(s2))]
    # part 3: 4 segments per match, a player present at all 10 sampled frames
    segs = []
    for m in ("117093", "118576", "132831"):
        pm = p[p.match == m]
        cand = []
        for (src, pid), x in pm.groupby(["src", "player_id"]):
            fr = set(x.frame)
            for f0 in sorted(fr):
                if all(f0 + STEP * j in fr for j in range(SEG_N)):  # matched at all 10 frames
                    cand.append((src, pid, f0))
        for j in rng.choice(len(cand), 4, replace=False):
            src, pid, f0 = cand[j]
            x = pm[(pm.src == src) & (pm.player_id == pid)].set_index("frame")
            segs.append([x.loc[f0 + STEP * k].to_dict() | {"frame": f0 + STEP * k, "src": src} for k in range(SEG_N)])
    order = rng.permutation(len(segs))
    s3 = pd.DataFrame([dict(r, seg=f"S{si:02d}", id=f"S{si:02d}-{k}") for si, oi in enumerate(order) for k, r in enumerate(segs[oi])])
    imgs = frames_needed(pd.concat([s2[["src", "frame"]], s3[["src", "frame"]]]))
    OUT.mkdir(exist_ok=True)
    c2 = [(r.id, crop(imgs[(r.src, r.frame)], r)) for r in s2.itertuples()]
    for i in range(0, len(c2), 24):
        sheet(c2[i:i + 24], 6, 300, 430, OUT / f"part2_{i // 24}.png")
    c3 = [(r.id, crop(imgs[(int(r.src), int(r.frame))], r)) for r in s3.itertuples()]
    for i in range(0, len(c3), 40):
        sheet(c3[i:i + 40], 10, 190, 330, OUT / f"part3_{i // 40}.png")
    key = {"part2": s2[["id", "match", "frame", "player_id", "jersey", "bh", "band"]].to_dict("records"),
           "part3": s3[["id", "seg", "match", "frame", "player_id", "jersey", "bh"]].to_dict("records")}
    (HERE / "key.json").write_text(json.dumps(key, default=str))
    print("part2", len(s2), s2.band.value_counts().to_dict(), "part3", len(s3), flush=True)


if __name__ == "__main__":
    main()
