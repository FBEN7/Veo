"""Score plan_v4 (+ addenda 1, 2): the tracker variants A, A2, B2 of the CPU
kernel veo-nearball-track (trk/<unit>_<v>.parquet) on the actions of
stages.parquet, at S1 (tracker, after the outline), S3 (team assigned) and
S4 (roster cap); reproduction checks against stages.parquet; B2 - A with a
bootstrap over matches; the bar.

    VEO_UNSEAL=... python score_v4.py TRK_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import eval_soccertrack_v2 as ev  # noqa: E402

CLIPS = ROOT / "data/shot_clips/soccertrack_v2"
SEALED = {"128057", "132831"}
NO_GRASS_TEST = {"128057", "132831"}        # addendum 1: tracked after ca98c6c2
UNASSIGNED = ("", "-1", "unknown", "unassigned", "nan", "None")
WIN = 24
# (variant, stage) -> (file, row filter)
CELLS = {("A", "S1"): ("A", "all"), ("A", "S3"): ("A", "team"), ("A", "S4"): ("A", "s4"),
         ("A2", "S3"): ("A2", "team"), ("A2", "S4"): ("A2", "s4"),
         ("B2", "S1"): ("B2", "all"), ("B2", "S3"): ("B2", "team"), ("B2", "S4"): ("B2", "s4")}


def rows_of(t: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "team":
        t = t[t.team.notna() & ~t.team.astype(str).isin(UNASSIGNED)]
    elif how == "s4":
        t = t[t.in_s4]
    return ev.boxes(t.assign(cls="player"))


def unit(w: str, trk: Path, acts: pd.DataFrame) -> list:
    gt = pd.read_parquet(CLIPS / f"{w}_gt.parquet")
    gt = gt[gt.bx.notna()]
    truth = ev.truth(CLIPS / f"{w}_gt.parquet")
    centres = sorted(set(acts.f))
    frames = sorted({c + k for c in centres for k in range(-WIN, WIN + 1, 2)})
    tr = truth[truth.frame.isin(frames)]
    tr_f = {f: g for f, g in tr.groupby("frame")}
    files = {v: pd.read_parquet(trk / f"{w}_{v}.parquet") for v in ("A", "A2", "B2")}
    out = []
    for (v, s), (fv, how) in CELLS.items():
        d = rows_of(files[fv], how)
        d = d[d.frame.isin(frames)].reset_index(drop=True)
        d["person"] = ev.match_feet(d, tr) if len(d) else pd.Series(dtype=float)
        by_f = {f: g for f, g in d.groupby("frame")}
        for r in acts.itertuples():
            a = gt[(gt.frame == r.f) & (gt.player_id.astype(str) == str(int(r.player_id)))]
            t = tr_f.get(r.f)
            if a.empty or t is None:
                continue
            actor = int(a.track_id.iloc[0])
            g = by_f.get(r.f, d.iloc[:0])
            au, av = float(a.bx.iloc[0] + a.bw.iloc[0] / 2), float(a.by.iloc[0] + a.bh.iloc[0])
            dist = np.hypot(((g.x0 + g.x1) / 2).to_numpy() - au, g.y1.to_numpy() - av)
            hit = g[g.person == actor]
            kind = "detected" if len(hit) else ("taken" if len(dist) and (dist <= ev.FOOT_PX).any() else "missed")
            others = t[(t.person != actor) & (t.role == "player")]
            ctrl = float(others.person.isin(set(g.person.dropna())).mean()) if len(others) else np.nan
            same = np.nan
            if kind == "detected" and s == "S4":
                tid = hit.track_id.iloc[0]
                win = [fr for fr in range(r.f - WIN, r.f + WIN + 1, 2) if fr in by_f]
                same = float(sum(((by_f[fr].person == actor) & (by_f[fr].track_id == tid)).any() for fr in win) / max(len(win), 1))
            out.append({"unit": w, "match": w.split("_")[1], "frame": r.frame, "player_id": int(r.player_id),
                        "variant": v, "stage": s, "kind": kind, "control": ctrl,
                        "unmatched": int(g.person.isna().sum()), "same track +-1s": same})
    return out


def boot(df: pd.DataFrame, a: str, b: str, stage: str = "S4") -> dict:
    x = df[df.stage == stage]
    matches = sorted(x.match.unique())
    det = {(m, v): int(((x.match == m) & (x.variant == v) & (x.kind == "detected")).sum()) for m in matches for v in (a, b)}
    n = {m: int(((x.match == m) & (x.variant == a)).sum()) for m in matches}
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(5000):
        s = rng.choice(matches, len(matches))
        diffs.append(sum(det[(m, b)] - det[(m, a)] for m in s) / sum(n[m] for m in s))
    gain = sum(det[(m, b)] - det[(m, a)] for m in matches) / sum(n.values())
    return {"gain": round(float(gain), 4),
            "95% bootstrap over matches": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
            "per match": {m: round((det[(m, b)] - det[(m, a)]) / n[m], 3) for m in matches}}


def main():
    from src.unseal import unsealed
    for m in sorted(SEALED):
        if not unsealed(m, "near_ball plan_v4: scoring the tracker variants on the opened M7 / M9 clips"):
            raise SystemExit("VEO_UNSEAL needed")
    trk = Path(sys.argv[1])
    st = pd.read_parquet(HERE / "stages.parquet")
    acts = st[st.stage == "S0 raw"][["unit", "frame", "player_id"]].drop_duplicates()
    acts["f"] = acts.frame
    rows, missing = [], []
    for i, (w, a) in enumerate(acts.groupby("unit")):
        if not all((trk / f"{w}_{v}.parquet").exists() for v in ("A", "A2", "B2")):
            missing.append(w)
            continue
        rows += unit(w, trk, a)
        if i % 25 == 0:
            print(i, w, flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(HERE / "v4.parquet")
    res = {}
    for (v, s), g in df.groupby(["variant", "stage"]):
        k = g.kind.value_counts(normalize=True)
        res[f"{v} {s}"] = {"actions": len(g), "detected": round(float(k.get("detected", 0)), 4),
                           "taken": round(float(k.get("taken", 0)), 4), "missed": round(float(k.get("missed", 0)), 4),
                           "control": round(float(g.control.mean()), 4),
                           "unmatched per action frame": round(float(g.drop_duplicates(["unit", "frame"]).unmatched.mean()), 3)}
        if s == "S4":
            res[f"{v} {s}"]["same track +-1s (detected actors)"] = round(float(g["same track +-1s"].mean()), 4)
    # reproduction checks (addendum 1): A against the original stages, by group
    key = ["unit", "frame", "player_id"]
    orig = st.pivot_table(index=key, columns="stage", values="kind", aggfunc="first")
    rep = {}
    for grp, sel in (("M7, M9 (no grass test in the original)", lambda m: m.isin(NO_GRASS_TEST)),
                     ("7 other matches (grass test in the original)", lambda m: ~m.isin(NO_GRASS_TEST))):
        o = orig[sel(orig.index.get_level_values("unit").str.split("_").str[1])]
        out = {}
        for s_new, s_old in (("S1", "S1 tracker"), ("S4", "S4 roster cap")):
            a = df[(df.variant == "A") & (df.stage == s_new)].set_index(key).kind
            j = o.join(a.rename("new"), how="inner")
            out[s_new] = {"original": round(float((j[s_old] == "detected").mean()), 4),
                          "A": round(float((j.new == "detected").mean()), 4), "actions": len(j)}
        out["S4 gap above 0.02"] = bool(abs(out["S4"]["A"] - out["S4"]["original"]) > 0.02)
        rep[grp] = out
    primary = boot(df, "A", "B2")
    a4, b4 = res["A S4"], res["B2 S4"]
    guards = {"unmatched per action frame, B2 <= 1.2 x A": bool(b4["unmatched per action frame"] <= 1.2 * a4["unmatched per action frame"]),
              "control, B2 >= A - 0.01": bool(b4["control"] >= a4["control"] - 0.01)}
    lo = primary["95% bootstrap over matches"][0]
    out = {"plan_v4 sha256": (HERE / "plan_v4.json.sha256").read_text().split()[0],
           "addendum1 sha256": (HERE / "plan_v4_addendum1.json.sha256").read_text().split()[0],
           "addendum2 sha256": (HERE / "plan_v4_addendum2.json.sha256").read_text().split()[0],
           "units missing": missing, "per variant and stage": res, "reproduction (A vs the original)": rep,
           "primary: actor detected at S4, B2 - A": primary,
           "decomposition: A2 - A": boot(df, "A", "A2"), "decomposition: B2 - A2": boot(df, "A2", "B2"),
           "guards": guards,
           "bar met (gain >= 0.03, lower bound > 0, guards)": bool(primary["gain"] >= 0.03 and lo > 0 and all(guards.values()))}
    (HERE / "v4.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
