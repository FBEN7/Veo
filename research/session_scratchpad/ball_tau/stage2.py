"""Stage 2 of plan.json for the chosen variant (zone03: the best share of
shots near a goal with the ball at the feet, 0.855 against 0.692, wrong
picks 0.071 against 0.069): the v3 shot spotter and rule C on its tracks.

    VEO_UNSEAL=... python stage2.py shots    # -> stage2_shots.json
    VEO_UNSEAL=... python stage2.py passer   # -> stage2_passer.json
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
for d in ("shots_learn", "shots_learn/v2", "shots_learn/v3", "ball_grass"):
    sys.path.insert(0, str(ROOT / "research/session_scratchpad" / d))
from src.unseal import unsealed  # noqa: E402

VARIANT = "zone03"
FNAME = f"metric_v3s_link_{VARIANT}.parquet"
NEW_PLAYERS = {"128057", "132831"}


def shots():
    import learn_shots as LS
    import learn_v2 as V
    import learn_v3 as V3
    tree = ROOT / f".cache/gpu_shots_{VARIANT}"
    us = []
    for u in V3.new_tree([u for u in V.units() if (LS.GPU / u["name"]).exists()]):   # the v3 units
        w = u["name"]
        src = ROOT / (".cache/gpu_nograss" if u["match"] in NEW_PLAYERS else ".cache/gpu") / w / f"output_{w}"
        if not (src / FNAME).exists():
            continue
        d = tree / w / f"output_{w}"
        d.mkdir(parents=True, exist_ok=True)
        for name, target in (("metric_v3s_link.parquet", src / FNAME), ("v3s_rows.parquet", src / "v3s_rows.parquet")):
            if not (d / name).exists():
                os.symlink(target, d / name)
        us.append(u)
    out = {"variant": VARIANT, "units": len(us), "result": V3.run(us, tree),
           "reference (shots_learn/v3/cv_v3.json, no grass tests)":
               json.loads((ROOT / "research/session_scratchpad/shots_learn/v3/cv_v3.json").read_text())["no grass tests"]}
    (HERE / "stage2_shots.json").write_text(json.dumps(out, indent=1))
    r = out["result"]
    print({k: r[k] for k in ("tp", "fp", "fn", "recall", "precision")})


def passer():
    import score as S
    seen = {"117093", "118575", "118576", "118577", "118578", "128058", "132877"}
    out = {"variant": VARIANT}
    for label, matches, root in (("seen matches", seen, ".cache/gpu"), ("M7 / M9", NEW_PLAYERS, ".cache/gpu_nograss")):
        us = [w for w in S.units_of(matches, root) if (ROOT / root / w / f"output_{w}" / FNAME).exists()]
        out[label] = {"units": len(us), "no grass (t10)": S.run(us, root, "metric_v3s_link_nograss.parquet"),
                      VARIANT: S.run(us, root, FNAME)}
        print(label, out[label]["no grass (t10)"]["all"]["share right"], out[label][VARIANT]["all"]["share right"], flush=True)
    (HERE / "stage2_passer.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    for m in NEW_PLAYERS:
        if not unsealed(m, f"ball_tau stage 2 ({sys.argv[1]})"):
            raise SystemExit("VEO_UNSEAL needed")
    {"shots": shots, "passer": passer}[sys.argv[1]]()
