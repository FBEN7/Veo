"""Why rule A names the wrong player (owner, 2026-10-26: "piste 1"): the
actions inside the clips' ball range on the matches v3 was trained on where
rule A names someone who is not the labelled player. Per case:

- chosen: the named detection's truth person -- none (no truth person:
  a false or merged detection), a teammate of the labelled player, or an
  opponent;
- labelled player detected at the label frame, and its distance to the
  ball against the named one's;
- timing: over -1 .. +1 s around the label (players' frames, the ball row
  nearest each within +-5 frames), the offsets at which the labelled
  player is the one nearest the ball within 3 m.

    python diagnose.py
"""
import json

import numpy as np
import pandas as pd

import passer as P

OFFS = range(-26, 27, 2)


def unit(w, lo, hi):
    info = json.loads((P.CLIPS / f"{w}.json").read_text())
    m = pd.read_parquet(P.GPU / w / f"output_{w}" / "metric_v3s_link.parquet")
    m[["x", "y"]] = P.st.image_to_pitch(m[["px_raw", "py_raw"]].to_numpy(), P.calibration(str(info["match"])))
    ball = m[m.cls == "ball"].set_index("frame")
    ball = ball[~ball.index.duplicated()]
    pl = m[m.cls == "player"]
    by_f = {f: g for f, g in pl.groupby("frame")}
    g = pd.read_parquet(P.CLIPS / f"{w}_gt.parquet")
    first = g.drop_duplicates("track_id").set_index("track_id")
    pid, side = first.player_id.astype(str), first.side
    person = P.ev.match_feet(P.ev.boxes(pl), P.ev.truth(P.CLIPS / f"{w}_gt.parquet"))
    who = person.map(pid)
    who_side = person.map(side)
    ks = np.array(sorted(by_f))

    def ball_at(f):
        near = ball.index[(ball.index >= f - 5) & (ball.index <= f + 5)]
        if not len(near):
            return None
        return ball.loc[near[np.argmin(abs(near - f))]]

    out = []
    for e in pd.read_csv(P.CLIPS / f"{w}_events.csv").itertuples():
        if e.label in P.NOT_ACTOR or pd.isna(e.player_id) or not (lo <= e.frame <= hi):
            continue
        truth = str(int(e.player_id))
        f = int(ks[np.argmin(abs(ks - e.frame))])
        b = ball_at(f)
        if b is None or f not in by_f:
            continue
        q = by_f[f]
        d = np.hypot(q.x - b.x, q.y - b.y)
        i = d.idxmin()
        if d[i] > P.RADIUS:
            continue
        r = {"unit": w, "match": info["match"], "abs": int(info["start_frame"]) + f, "label": e.label,
             "right": who.get(i) == truth, "d_named": round(float(d[i]), 2)}
        if r["right"]:
            out.append(r)
            continue
        r["named"] = ("none" if pd.isna(who.get(i)) else
                      "teammate" if who_side.get(i) == e.side else "opponent")
        mine = q.index[(who[q.index] == truth).to_numpy()]
        r["labelled detected"] = len(mine) > 0
        r["d_labelled"] = round(float(d[mine].min()), 2) if len(mine) else None
        hits = []
        for o in OFFS:
            ff = f + o
            if ff not in by_f or (bb := ball_at(ff)) is None:
                continue
            qq = by_f[ff]
            dd = np.hypot(qq.x - bb.x, qq.y - bb.y)
            j = dd.idxmin()
            if dd[j] <= P.RADIUS and who.get(j) == truth:
                hits.append(o)
        r["nearest at offsets"] = hits
        out.append(r)
    return out


def main():
    units = sorted(p.name for p in P.GPU.iterdir() if (P.CLIPS / f"{p.name}_events.csv").exists()
                   and (P.GPU / p.name / f"output_{p.name}" / "metric_v3s_link.parquet").exists()
                   and p.name.split("_")[1] not in P.UNSEEN)
    rows = []
    for w in units:
        fr = json.loads((P.GPU / w / ".cache" / f"v3_balls_{w}.parquet.meta.json").read_text())["frames"]
        rows += unit(w, min(fr), max(fr))
    df = pd.DataFrame(rows).drop_duplicates(["match", "abs", "label"])
    wrong = df[~df.right].copy()
    wrong["timing"] = wrong["nearest at offsets"].map(
        lambda h: "never in +-1 s" if not h else "before" if max(h) < 0 else "after" if min(h) > 0 else "around")
    near = wrong[wrong["labelled detected"]]
    out = {
        "named (rule A, in range, seen matches)": len(df), "right": int(df.right.sum()), "wrong": len(wrong),
        "wrong: who was named": wrong.named.value_counts().to_dict(),
        "wrong: labelled player detected at the frame": round(wrong["labelled detected"].mean(), 3),
        "wrong and detected: labelled player's distance to the ball (m), quartiles":
            near.d_labelled.quantile([.25, .5, .75]).round(2).tolist(),
        "wrong and detected: named player's distance (m), quartiles": near.d_named.quantile([.25, .5, .75]).round(2).tolist(),
        "wrong and detected: labelled within 1 m of the named": round(((near.d_labelled - near.d_named) <= 1.0).mean(), 3),
        "wrong: when the labelled player is the nearest in +-1 s": wrong.timing.value_counts().to_dict(),
        "wrong, by label": wrong.label.value_counts().to_dict(),
        "right: named player's distance (m), quartiles": df[df.right].d_named.quantile([.25, .5, .75]).round(2).tolist(),
    }
    hits = [o for h in wrong["nearest at offsets"] for o in h]
    out["wrong: offsets (frames) where the labelled player is nearest, histogram"] = \
        pd.Series(hits).value_counts().sort_index().to_dict()
    print(json.dumps(out, indent=1, default=str))
    (P.HERE / "diagnose.json").write_text(json.dumps(out, indent=1, default=str))
    df.to_parquet(P.HERE / "diagnose_rows.parquet")


if __name__ == "__main__":
    main()
