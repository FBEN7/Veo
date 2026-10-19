"""Score match-wide player names against SoccerNet game-state ground truth.

Clips of one game in the game-state set name every person by team side
and shirt number, so a person is the same across clips when both agree
(sides swap at half-time). Each clip's identities are matched to the real
person they mostly follow (`eval_identity_gsr.match`); people whose number
is never legible cannot be followed across clips and are left out.

Over pairs of identities from *different* clips, both of a known person:

- **precision**: of the pairs given one match-wide name, the share that are
  one person;
- **recall**: of the pairs that are one person, the share given one name;
- the team alignment, checked against the ground truth's sides.

The threshold is chosen on `--fit` clips and reported on `--test` clips.

    python eval_match_identity.py --fit output_gsr_SNGS-021 ... \\
        --test output_gsr_SNGS-026 ...
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

from eval_identity_gsr import match, ours
from src import match_identity as mi
from src.track_split import player_tracks

THRESHOLDS = (0.9, 0.8, 0.7, 0.6, 0.5)


def person_keys(out_dir: Path, gsr: Path) -> dict:
    """identity key -> (side, jersey) of the person it mostly follows."""
    name = out_dir.name.replace("output_gsr_", "")
    blob = json.loads((gsr / name / "Labels-GameState.json").read_text())
    half = int(blob["info"]["game_time_start"].split(" - ")[0])
    frame_of = {im["image_id"]: int(Path(im["file_name"]).stem) - 1
                for im in blob["images"]}
    rows = []
    for a in blob["annotations"]:
        if a.get("category_id") not in (1, 2) or "bbox_image" not in a:
            continue
        att = a.get("attributes") or {}
        side, jersey = att.get("team"), att.get("jersey")
        if half == 2 and side in ("left", "right"):
            side = "right" if side == "left" else "left"
        b = a["bbox_image"]
        rows.append({"frame": frame_of[a["image_id"]], "person": a["track_id"],
                     "x0": b["x"], "y0": b["y"], "x1": b["x"] + b["w"],
                     "y1": b["y"] + b["h"], "side": side,
                     "jersey": jersey if jersey not in (None, "") else None})
    gt = pd.DataFrame(rows)
    key_of = gt.groupby("person").apply(
        lambda g: (g.side.mode().iloc[0] if g.side.notna().any() else None,
                   g.jersey.dropna().mode().iloc[0]
                   if g.jersey.notna().any() else None))
    tracks = player_tracks(out_dir)
    det = ours(tracks)
    person = match(det, gt)
    ident = json.loads((out_dir / "identities.json").read_text())["tracks"]
    who = det.track_id.map(lambda t: ident.get(str(int(t)), {}).get("identity"))
    df = pd.DataFrame({"ident": who, "person": person}).dropna()
    main = df.groupby("ident").person.agg(lambda s: s.mode().iloc[0])
    out = {}
    for k, p in main.items():
        side, jersey = key_of.get(p, (None, None))
        if side and jersey:
            out[int(k)] = (side, str(jersey))
    return out


def score(dirs, threshold, keys):
    rows = mi.load(dirs)
    names = mi.link(rows, threshold)
    known = [((r["clip"], r["key"]), keys[r["clip"]].get(r["key"]))
             for r in rows if keys[r["clip"]].get(r["key"])
             and r["frames"] >= mi.MIN_FRAMES]
    tp = fp = fn = 0
    for (a, ka), (b, kb) in itertools.combinations(known, 2):
        if a[0] == b[0]:
            continue
        same_name = names[a] == names[b]
        same = ka == kb
        tp += same and same_name
        fp += same_name and not same
        fn += same and not same_name
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--gsr", default=".cache/gsr")
    args = ap.parse_args()
    keys = {Path(d).name: person_keys(Path(d), Path(args.gsr))
            for d in args.fit + args.test}
    # Team alignment against the ground truth's sides.
    rows = mi.load(args.fit + args.test)
    teams = mi.align_teams(rows)
    agree = {}
    for r in rows:
        k = keys[r["clip"]].get(r["key"])
        if k and r["team"] in mi.TEAMS:
            agree.setdefault((teams[r["clip"]][r["team"]], k[0]), 0)
            agree[(teams[r["clip"]][r["team"]], k[0])] += 1
    print("  aligned team (ours) vs side (truth), identities:", agree)
    best = None
    for thr in THRESHOLDS:
        tp, fp, fn = score(args.fit, thr, keys)
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f"  fit, threshold {thr}: precision {p:.0%}, recall {r:.0%} "
              f"({tp} right links, {fp} wrong, {fn} missed)")
        if best is None or f1 > best[1]:
            best = (thr, f1)
    tp, fp, fn = score(args.test, best[0], keys)
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    print(f"\n  test, threshold {best[0]}: precision {p:.0%}, recall {r:.0%} "
          f"({tp} right links, {fp} wrong, {fn} missed)")


if __name__ == "__main__":
    main()
