"""Rule C, possession (owner, 2026-10-26): over a window before the label,
at every players' frame with a ball row within +-2 frames, the player
nearest the ball within 3 m gets a vote; the track with most votes is the
actor (ties: the vote nearest the label), its detection nearest the label
is scored as passer.py. Windows tried (frames): [-15, +3], [-25, +5],
[-10, +5]; A and B recomputed on the same actions. Inside the clips' ball
range, matches v3 was trained on; exploratory, no held-out match (M1 not
measurable), so the window choice is in-sample.

    python possession.py
"""
import json

import numpy as np
import pandas as pd

import passer as P

WINDOWS = {"C [-0.6, +0.12] s": (-15, 3), "C [-1.0, +0.2] s": (-25, 5), "C [-0.4, +0.2] s": (-10, 5)}


def unit(w, lo, hi):
    info = json.loads((P.CLIPS / f"{w}.json").read_text())
    m = pd.read_parquet(P.GPU / w / f"output_{w}" / "metric_v3s_link.parquet")
    m[["x", "y"]] = P.st.image_to_pitch(m[["px_raw", "py_raw"]].to_numpy(), P.calibration(str(info["match"])))
    ball = m[m.cls == "ball"].set_index("frame")
    ball = ball[~ball.index.duplicated()].sort_index()
    pl = m[m.cls == "player"]
    by_f = {f: g for f, g in pl.groupby("frame")}
    pid = pd.read_parquet(P.CLIPS / f"{w}_gt.parquet").drop_duplicates("track_id").set_index("track_id").player_id.astype(str)
    who = P.ev.match_feet(P.ev.boxes(pl), P.ev.truth(P.CLIPS / f"{w}_gt.parquet")).map(pid)
    ks = np.array(sorted(by_f))
    bi = ball.index.to_numpy()

    def nearest_at(f, tol):
        if f not in by_f or not len(bi):
            return None
        j = np.argmin(abs(bi - f))
        if abs(bi[j] - f) > tol:
            return None
        b = ball.iloc[j]
        q = by_f[f]
        d = np.hypot(q.x - b.x, q.y - b.y)
        i = d.idxmin()
        return (i, float(d[i])) if d[i] <= P.RADIUS else None

    out = []
    for e in pd.read_csv(P.CLIPS / f"{w}_events.csv").itertuples():
        if e.label in P.NOT_ACTOR or pd.isna(e.player_id) or not (lo <= e.frame <= hi):
            continue
        truth = str(int(e.player_id))
        f = int(ks[np.argmin(abs(ks - e.frame))])
        r = {"match": info["match"], "abs": int(info["start_frame"]) + f, "label": e.label}
        a = nearest_at(f, 5)
        r["A"] = None if a is None else who.get(a[0]) == truth
        best = None
        for ff in range(f - 10, f + 6):
            x = nearest_at(ff, 0)
            if x and (best is None or x[1] < best[1]):
                best = x
        r["B"] = None if best is None else who.get(best[0]) == truth
        for name, (s, t) in WINDOWS.items():
            votes = []
            for ff in range(f + s, f + t + 1):
                x = nearest_at(ff, 2)
                if x:
                    votes.append((int(pl.track_id[x[0]]), abs(ff - f), x[0]))
            if not votes:
                r[name] = None
                continue
            v = pd.DataFrame(votes, columns=["track", "lag", "idx"])
            c = v.groupby("track").agg(n=("lag", "size"), lag=("lag", "min"))
            top = c.sort_values(["n", "lag"], ascending=[False, True]).index[0]
            idx = v[v.track == top].sort_values("lag").idx.iloc[0]
            r[name] = who.get(idx) == truth
        out.append(r)
    return out


def summary(df, col):
    n = len(df)
    given = df[col].notna()
    right = int((df[col] == True).sum())  # noqa: E712
    return {"n": n, "right": right, "share right": round(right / n, 3), "wilson": P.wilson(right, n),
            "coverage": round(given.mean(), 3), "precision": round(right / max(given.sum(), 1), 3)}


def main():
    units = sorted(p.name for p in P.GPU.iterdir() if (P.CLIPS / f"{p.name}_events.csv").exists()
                   and (P.GPU / p.name / f"output_{p.name}" / "metric_v3s_link.parquet").exists()
                   and p.name.split("_")[1] not in P.UNSEEN)
    rows = []
    for w in units:
        fr = json.loads((P.GPU / w / ".cache" / f"v3_balls_{w}.parquet.meta.json").read_text())["frames"]
        rows += unit(w, min(fr), max(fr))
    df = pd.DataFrame(rows).drop_duplicates(["match", "abs", "label"])
    cols = ["A", "B", *WINDOWS]
    out = {"all actions": {c: summary(df, c) for c in cols},
           "passes": {c: summary(df[df.label.isin(P.PASSES)], c) for c in cols}}
    best = max(WINDOWS, key=lambda c: out["all actions"][c]["share right"])
    both = df[df.B.notna() & df[best].notna()]
    out[f"paired, {best} vs B"] = {"n": len(both), "C right B wrong": int(((both[best] == True) & (both.B == False)).sum()),  # noqa: E712
                                   "B right C wrong": int(((both.B == True) & (both[best] == False)).sum())}  # noqa: E712
    print(json.dumps(out, indent=1, default=str))
    (P.HERE / "possession.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
