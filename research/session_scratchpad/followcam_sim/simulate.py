"""follow-cam ceiling (plan.json, hashed first): a pinhole camera at the
halfway line follows a ball proxy built from SoccerTrack v2's actions; what
the report needs that stays in view. CPU only, truth only, no video.

    python simulate.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from src import soccertrack_v2 as st  # noqa: E402

FPS, STEP, NEAR_M, NEXT_S = 25, 5, 4.0, 6.0
DEAD_S = 6.0   # possession_v2 truth(): a gap > 6 s, or after OUT, is dead
BACK_M, HEIGHT_M, TALL_M, OUT_H = 3.0, 6.0, 1.8, 1080
FOVS = (30, 45, 60)
HALVES = [("117092", 1)] + [(m, h) for m in ("117093", "118575", "118576", "118577", "118578", "128058", "132877") for h in (1, 2)]
PASSES = ("PASS", "HIGH PASS", "CROSS")


def check():
    want = (HERE / "plan.json.sha256").read_text().split()[0]
    got = hashlib.sha256((HERE / "plan.json").read_bytes()).hexdigest()
    if got != want:
        raise SystemExit("plan.json does not match its sha256")
    return got


def proxy(ev, n):
    """ball proxy (x, y) per frame, NaN where not scored: the actor's position
    interpolated between actions over the stretches truth() counts as live."""
    e = ev[ev.side.isin(["left", "right"])].sort_values("frame", kind="stable")
    f, lab = e.frame.to_numpy(), e.label.to_numpy()
    xy = e[["x", "y"]].to_numpy(float)
    out = np.full((n, 2), np.nan)
    for i in range(len(f) - 1):
        a, b = int(f[i]), int(min(f[i + 1], n))
        if lab[i] == "OUT" or (b - a) / FPS > DEAD_S or b <= a or np.isnan(xy[i]).any() or np.isnan(xy[i + 1]).any():
            continue
        t = (np.arange(a, b) - f[i]) / max(f[i + 1] - f[i], 1)
        out[a:b] = xy[i] + t[:, None] * (xy[i + 1] - xy[i])
    return out


def aim(px):
    """camera target: centred 1-s rolling mean of the proxy (scored frames only)."""
    d = pd.DataFrame(px)
    return d.rolling(25, center=True, min_periods=1).mean().to_numpy() * np.where(np.isnan(px), np.nan, 1.0)


def project(cam, target, pts, fov):
    """pts (..., 3) seen from cam aimed at target (ground point): (in view, image y in px of a 1080 frame)."""
    f = np.append(target, 0.0) - cam
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    v = pts - cam
    zc, xc, yc = v @ f, v @ r, v @ u
    tx = np.tan(np.radians(fov) / 2)
    ty = tx * 9 / 16
    with np.errstate(divide="ignore", invalid="ignore"):
        inside = (zc > 0) & (np.abs(xc / zc) <= tx) & (np.abs(yc / zc) <= ty)
        img_y = (yc / zc) / ty * (OUT_H / 2)
    return inside, img_y


def per_half(match, half):
    gsr = st.gsr_table(match, half)
    gsr = gsr[gsr.role.isin(["player", "goalkeeper"]) & gsr.x.notna() & gsr.y.notna()]
    n = int(gsr.frame.max()) + 1
    ev = st.events(match)
    ev = st.with_actors(ev[ev.half == half], gsr)
    cy = (34.0 + BACK_M) * (1 if st.y_towards_camera(gsr) else -1)
    cam = np.array([0.0, cy, HEIGHT_M])
    px = proxy(ev, n)
    tg = aim(px)
    scored = np.where(~np.isnan(tg[:, 0]))[0]
    samp = scored[::STEP]
    G = {f: g for f, g in gsr[gsr.frame.isin(set(samp))].groupby("frame")}
    out = {}
    # per-frame measures
    for fov in FOVS:
        poss_ok = poss_n = poss_none = 0
        in_view, sizes = [], []
        for fr in samp:
            g = G.get(fr)
            if g is None:
                continue
            pts = np.c_[g.x.to_numpy(float), g.y.to_numpy(float), np.zeros(len(g))]
            ins, y0 = project(cam, tg[fr], pts, fov)
            _, y1 = project(cam, tg[fr], pts + [0, 0, TALL_M], fov)
            in_view.append(int(ins.sum()))
            sizes += list(np.abs(y0 - y1)[ins])
            near = np.hypot(pts[:, 0] - px[fr, 0], pts[:, 1] - px[fr, 1]) <= NEAR_M
            if not near.any():
                poss_none += 1
                continue
            poss_n += 1
            poss_ok += int(ins[near].all())
        out[fov] = {"possession frames": poss_n, "possession ok": poss_ok, "no one within 4 m": poss_none,
                    "frames": len(in_view), "in view sum": int(np.sum(in_view)), "sizes": sizes}
    # passes
    e = ev[ev.side.isin(["left", "right"])].sort_values("frame", kind="stable").reset_index(drop=True)
    pos = gsr.set_index(["frame", "player_id"])[["x", "y"]]
    passes = []
    for i in range(len(e) - 1):
        a, b = e.iloc[i], e.iloc[i + 1]
        if a.label not in PASSES or (b.frame - a.frame) / FPS > NEXT_S or b.frame >= n:
            continue
        if np.isnan(tg[int(a.frame), 0]) or np.isnan(tg[int(b.frame), 0]):
            continue
        try:
            pa = pos.loc[(int(a.frame), a.player_id)].to_numpy(float).ravel()[:2]
            pb = pos.loc[(int(b.frame), b.player_id)].to_numpy(float).ravel()[:2]
        except KeyError:
            continue
        row = {"length": float(np.hypot(*(pa - pb)))}
        for fov in FOVS:
            ia, _ = project(cam, tg[int(a.frame)], np.array([[pa[0], pa[1], 0.0]]), fov)
            ib, _ = project(cam, tg[int(b.frame)], np.array([[pb[0], pb[1], 0.0]]), fov)
            row[fov] = bool(ia[0] and ib[0])
        passes.append(row)
    # distances: positions smoothed over 1 s, sampled each second
    secs = np.arange(12, n, 25)
    dist = {fov: [] for fov in FOVS}
    team = {fov: {} for fov in FOVS}
    for (pid, side), d in gsr.groupby(["player_id", "side"]):
        d = d.drop_duplicates("frame").set_index("frame").sort_index()[["x", "y"]]
        sm = d.reindex(np.arange(n)).rolling(25, center=True, min_periods=25).mean().reindex(secs).to_numpy()
        seg = np.hypot(*np.diff(sm, axis=0).T)
        ok = ~np.isnan(seg) & ~np.isnan(tg[secs[:-1], 0]) & ~np.isnan(tg[secs[1:], 0])
        if ok.sum() < 60:
            continue
        for fov in FOVS:
            vis = np.zeros(len(secs), bool)
            for k, fr in enumerate(secs):
                if not np.isnan(tg[fr, 0]) and not np.isnan(sm[k, 0]):
                    vis[k] = project(cam, tg[fr], np.array([[sm[k, 0], sm[k, 1], 0.0]]), fov)[0][0]
            seen = ok & vis[:-1] & vis[1:]
            tot, sn = float(seg[ok].sum()), float(seg[seen].sum())
            dist[fov].append(sn / tot if tot else np.nan)
            t = team[fov].setdefault(side, [0.0, 0.0])
            t[0] += sn
            t[1] += tot
    return out, passes, dist, {fov: [s / t for s, t in team[fov].values() if t] for fov in FOVS}, len(scored) / n


def main():
    sha = check()
    agg = {fov: {"possession frames": 0, "possession ok": 0, "no one within 4 m": 0, "frames": 0, "in view sum": 0, "sizes": []} for fov in FOVS}
    passes, dist, team, cover = [], {fov: [] for fov in FOVS}, {fov: [] for fov in FOVS}, {}
    for match, half in HALVES:
        o, p, d, t, c = per_half(match, half)
        cover[f"{match} {half}"] = round(c, 3)
        for fov in FOVS:
            for k in ("possession frames", "possession ok", "no one within 4 m", "frames", "in view sum"):
                agg[fov][k] += o[fov][k]
            agg[fov]["sizes"] += o[fov]["sizes"]
            dist[fov] += d[fov]
            team[fov] += t[fov]
        passes += p
        print(match, half, "done", flush=True)
    pp = pd.DataFrame(passes)
    pp["band"] = pd.cut(pp.length, [0, 15, 30, 1e9], right=False, labels=["< 15 m", "15-30 m", ">= 30 m"])
    res = {}
    for fov in FOVS:
        a = agg[fov]
        dd = np.array(dist[fov], float)
        res[f"{fov} deg"] = {
            "possession: players within 4 m all in view": round(a["possession ok"] / a["possession frames"], 3),
            "players in view per frame (of 22)": round(a["in view sum"] / a["frames"], 1),
            "passes: passer and next actor in view": round(float(pp[fov].mean()), 3),
            "passes by length": {str(b): round(float(g[fov].mean()), 3) for b, g in pp.groupby("band", observed=True)},
            "player distance: median share of path in view": round(float(np.nanmedian(dd)), 3),
            "player distance: players with >= 95% in view": round(float(np.nanmean(dd >= 0.95)), 3),
            "team distance: median share of path in view": round(float(np.median(team[fov])), 3),
            "player height in a 1080-px output, median px (upper bound)": round(float(np.median(a["sizes"])), 1),
        }
    out = {"plan sha256": sha, "halves": len(HALVES), "scored share of frames per half": cover,
           "scored frames (sampled)": agg[FOVS[0]]["frames"], "frames with no one within 4 m of the proxy": agg[FOVS[0]]["no one within 4 m"],
           "passes scored": len(pp), "passes by length": pp.band.value_counts().sort_index().to_dict(),
           "players scored (player-halves)": len(dist[FOVS[0]]), "results": res}
    (HERE / "followcam_sim.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
