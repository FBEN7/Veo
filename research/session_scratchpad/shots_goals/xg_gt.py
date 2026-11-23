"""xG on ground truth: the frozen xGHub model on GSR shooter feet (protocol XG).

Implements the XG section of the registered protocol
(research/session_scratchpad/shots_goals/protocol.json, checked against its
sha256 before anything is read): "descriptive only: the frozen xGHub model
(models/xg_xghub.json) on GSR shooter feet for all BAS SHOT/GOAL with an actor
position; AUC with a half-block bootstrap, Brier against the base rate,
quintile calibration; M1 shown apart. No threshold. Penalties cannot be told
apart (no source)."

Method, per half of the 10 matches (st.MATCHES):
  1. Attempts: every BAS action labelled SHOT or GOAL (st.events; the third
     period some matches played is already dropped there). Outcome: GOAL = 1,
     SHOT = 0.
  2. Shooter feet: st.with_actors on that half's GSR table (the record with
     the action's player_id on the action's frame), centre-origin metres.
     Halves with no cached GSR parquet are skipped and listed (nothing is
     streamed). Attempts with no player_id or no matching GSR record, or a
     record with no pitch position, are dropped and counted.
  3. Attacked goal: the shooter's side is the side of his own GSR record on
     that frame. Each side's people (all its GSR rows over the half: players
     and goalkeeper; GSR has no side for anyone else) have a mean x; the side
     with the lower mean x defends the left end (corner-origin x = 0) and
     attacks "right", the other defends the right end and attacks "left".
  4. Geometry: corner-origin (x + 52.5, y + 34) into
     shot_geometry.shot_features with that goal (called directly with x, y:
     shot_features_from_events reads shot['x'], which BAS rows do not carry).
     xG: xg.score_shots with the frozen model; no attempt is a penalty
     (is_penalty False for all: nothing marks one).
  5. Report, for all matches, all but M1 and M1 (117092) alone: n, goals,
     sum and mean xG; AUC with a half-block bootstrap (halves drawn with
     replacement, as many as the set has, all their attempts pooled, 1000
     draws, seed 0, draws with one class only are skipped and counted, 95%
     percentile interval); Brier against the base rate; the quintile
     calibration table (xg.calibration_table, bins=5).

Choices where the protocol is silent (simplest reading):
  - "The base rate" is the set's own goal share (the best constant forecast
    in sample), so Brier skill = 1 - Brier / (p (1 - p)). The xGHub training
    base rate (models/xg_xghub.json, 0.117) is reported beside it.
  - "M1 shown apart": M1 alone, plus all matches and all but M1. M1's
    bootstrap has only 2 blocks, so its interval says little; it is reported
    because the protocol asks for the same treatment.
  - "Its mean x over the half" is read as a comparison of the two sides'
    means (lower mean x: defends the left end). This was not the first
    reading. The first run read it as the sign of each side's own mean x
    against the halfway line, and that reading put both sides on the same
    end in 6 of the 18 halves (one team camped in the other's half; e.g.
    118576 1st, both means < 0, so one team "attacked" its own goal): 87 of
    268 attempts came from beyond halfway, AUC 0.555. The protocol's "the
    end it defends" presupposes the two teams defend opposite ends, which
    only the comparison guarantees; on the 14 non-test halves it agrees with
    the goalkeepers' mean x in every half. The sign reading is still
    computed and written under "superseded: sign reading" so the switch is
    on the record; it is not the result. The same ambiguity sits in the
    GOALS section ("A team's own half is the end it defends: its mean x
    over the half").
  - Extra, not in the protocol: a half-block bootstrap interval of the Brier
    skill (same draws), and diagnostics (BAS vs GSR shooter side, attempts
    from beyond halfway, GOALs with a SHOT by the same player in the 5 s
    before, duplicate GSR records).
  - M1's GSR positions are the release's own pitch annotations; the M1
    calibration override is for detected feet only and is not used here.

Test hygiene: this is the one script allowed to read GSR at SHOT/GOAL frames
of the test halves (117092 both, 132831 2nd, 118577 2nd). For those halves it
uses only the shooter rows at the action frames and each side's half-mean x,
and it writes no per-shot rows, no per-half row and no defended end: they
enter the JSON only inside the pooled numbers (and M1's). It reads no video
and no detections.

    PYTHONPATH=/home/user/Veo python research/session_scratchpad/shots_goals/xg_gt.py
    -> research/session_scratchpad/shots_goals/xg_gt.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/Veo")
sys.path.insert(0, str(ROOT))
from src import shot_geometry as sg  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src import xg  # noqa: E402

HERE = ROOT / "research/session_scratchpad/shots_goals"
PROTOCOL = HERE / "protocol.json"
MODEL = ROOT / "models/xg_xghub.json"
OUT = HERE / "xg_gt.json"
M1 = "117092"
LABELS = ("SHOT", "GOAL")
HALF_NAMES = {1: "1st", 2: "2nd"}
DRAWS = 1000
SEED = 0
GOAL_GAP_S = 5.0     # diagnostic: a SHOT by the scorer this soon before a GOAL


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    """The registered protocol, refused if it no longer matches its hash."""
    want = (HERE / "protocol.json.sha256").read_text().split()[0]
    got = sha256(PROTOCOL)
    if got != want:
        raise SystemExit(f"protocol.json sha256 {got} != registered {want}")
    return json.loads(PROTOCOL.read_text())


def gsr_cached(match: str, half: int) -> bool:
    return (st.cache_dir() / "gsr" / f"{match}_{HALF_NAMES[half]}.parquet").exists()


def defended_end(mean_x: dict, rule: str) -> dict:
    """The end each GSR side defends, from its mean x over the half.
    "compare" (the result): the lower mean x defends the left end
    (corner-origin x = 0). "sign" (superseded): mean x < 0 defends the left
    end, each side on its own."""
    if rule == "sign":
        return {s: ("left" if m < 0 else "right") for s, m in mean_x.items()}
    if len(mean_x) != 2:
        raise ValueError(f"expected two GSR sides, got {sorted(mean_x)}")
    low = min(mean_x, key=mean_x.get)
    return {s: ("left" if s == low else "right") for s in mean_x}


def shooters_of_half(match: str, half: int, ev: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """The half's SHOT/GOAL attempts with shooter feet and GSR side, each
    side's mean x over the half, and the half's diagnostics."""
    shots = ev[(ev.half == half) & ev.label.isin(LABELS)].reset_index(drop=True)
    gsr = st.gsr_table(match, half)
    keyed = gsr[gsr.player_id.notna()]
    n_dup = int(keyed.duplicated(["frame", "player_id"]).sum())
    gsr = pd.concat([keyed.drop_duplicates(["frame", "player_id"]), gsr[gsr.player_id.isna()]])
    a = st.with_actors(shots, gsr)
    side = gsr[gsr.player_id.notna()][["frame", "player_id", "side"]].rename(columns={"side": "gsr_side"})
    a = a.merge(side, on=["frame", "player_id"], how="left")
    mean_x = gsr[gsr.side.notna()].groupby("side").x.mean().to_dict()
    has = a.x.notna() & a.y.notna() & a.gsr_side.isin(list(mean_x))
    a = a[has].reset_index(drop=True)
    a["is_goal"] = (a.label == "GOAL").astype(int)
    a["match"], a["block"] = match, f"{match}_{HALF_NAMES[half]}"
    diag = {"bas attempts": len(shots), "no actor position": int((~has).sum()),
            "duplicate gsr records": n_dup, "side bas != gsr": int((a.side != a.gsr_side).sum()),
            "goal after own shot": goals_after_own_shot(shots)}
    return a, mean_x, diag


def with_geometry(a: pd.DataFrame, end: dict) -> pd.DataFrame:
    """Attacked goal (opposite the shooter's side's end) and the shot
    geometry from corner-origin feet."""
    a = a.copy()
    a["goal"] = ["right" if end[s] == "left" else "left" for s in a.gsr_side]
    a["xc"], a["yc"] = a.x + sg.PITCH_LENGTH_M / 2, a.y + sg.PITCH_WIDTH_M / 2
    rows = [sg.shot_features(r.xc, r.yc, r.goal) for r in a.itertuples()]
    for k in ("distance_m", "angle_rad", "forward_m", "lateral_m"):
        a[k] = [float(f[k]) for f in rows]
    return a


def goals_after_own_shot(shots: pd.DataFrame) -> int:
    """GOALs with a SHOT by the same player in the GOAL_GAP_S before it (a
    possible double label of one attempt)."""
    n = 0
    for g in shots[shots.label == "GOAL"].itertuples():
        s = shots[(shots.label == "SHOT") & (shots.player_id == g.player_id)
                  & (shots.frame < g.frame) & (shots.frame >= g.frame - GOAL_GAP_S * st.FPS)]
        n += int(len(s) > 0)
    return n


def auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def brier_skill(y: np.ndarray, p: np.ndarray) -> float:
    rate = y.mean()
    base = rate * (1 - rate)
    return float(1 - np.mean((p - y) ** 2) / base) if base > 0 else float("nan")


def block_bootstrap(df: pd.DataFrame) -> dict:
    """AUC (and Brier skill) over halves drawn with replacement."""
    blocks = sorted(df.block.unique())
    by = {b: (g.is_goal.to_numpy(float), g.xg.to_numpy(float)) for b, g in df.groupby("block")}
    rng = np.random.default_rng(SEED)
    aucs, skills, skipped = [], [], 0
    for _ in range(DRAWS):
        pick = rng.integers(0, len(blocks), len(blocks))
        y = np.concatenate([by[blocks[i]][0] for i in pick])
        p = np.concatenate([by[blocks[i]][1] for i in pick])
        if len(np.unique(y)) < 2:
            skipped += 1
            continue
        aucs.append(auc(y, p))
        skills.append(brier_skill(y, p))

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None

    return {"blocks": len(blocks), "draws": DRAWS, "seed": SEED, "one-class draws skipped": skipped,
            "auc ci95": ci(aucs), "brier skill ci95 (extra)": ci(skills)}


def summary(df: pd.DataFrame, base_xghub: float) -> dict:
    y, p = df.is_goal.to_numpy(float), df.xg.to_numpy(float)
    rate = float(y.mean())
    brier = float(np.mean((p - y) ** 2))
    cal = xg.calibration_table(y, p, bins=5)
    return {
        "n": int(len(df)), "goals": int(y.sum()), "base rate": rate,
        "xg sum": float(p.sum()), "xg mean": float(p.mean()),
        "auc": auc(y, p), "bootstrap over halves": block_bootstrap(df),
        "brier": brier, "brier base rate": rate * (1 - rate), "brier skill": brier_skill(y, p),
        "brier xghub base rate": float(np.mean((base_xghub - y) ** 2)),
        "median distance m": {lab: float(df[df.label == lab].distance_m.median()) for lab in LABELS},
        "quintile calibration": cal.to_dict(orient="records"),
    }


def beyond_halfway(df: pd.DataFrame) -> int:
    """Attempts more than half a pitch from the goal they attack."""
    return int((df.forward_m > sg.PITCH_LENGTH_M / 2).sum())


def superseded(df: pd.DataFrame, one_end: int, base_xghub: float) -> dict:
    """The headline numbers under the first (sign) reading, for the record."""
    s = summary(df, base_xghub)
    b = s["bootstrap over halves"]
    return {"note": "first reading, superseded: each side's own mean x sign against halfway",
            "halves with both sides on one end": one_end, "beyond halfway": beyond_halfway(df),
            "n": s["n"], "goals": s["goals"], "auc": s["auc"], "auc ci95": b["auc ci95"],
            "brier": s["brier"], "brier skill": s["brier skill"]}


def run() -> dict:
    proto = protocol()
    test = set(proto["GOALS"]["test halves"])
    model = xg.XGModel.load(MODEL)
    parts, old, per_half, skipped, diag_sum, one_end = [], [], [], [], {}, 0
    for match in st.MATCHES:
        ev = st.events(match)
        for half in (1, 2):
            name = f"{match}_{HALF_NAMES[half]}"
            if not gsr_cached(match, half):
                sel = ev[(ev.half == half) & ev.label.isin(LABELS)]
                skipped.append({"half": name, **{lab: int((sel.label == lab).sum()) for lab in LABELS}})
                continue
            a, mean_x, diag = shooters_of_half(match, half, ev)
            for k, v in diag.items():
                diag_sum[k] = diag_sum.get(k, 0) + v
            sign_end = defended_end(mean_x, "sign")
            one_end += int(len(set(sign_end.values())) < len(sign_end))
            old.append(xg.score_shots(model, with_geometry(a, sign_end), verbose=False))
            a = xg.score_shots(model, with_geometry(a, defended_end(mean_x, "compare")), verbose=False)
            parts.append(a)
            if name not in test:
                per_half.append({"half": name, "n": int(len(a)), "goals": int(a.is_goal.sum()),
                                 "xg sum": float(a.xg.sum())})
    df = pd.concat(parts, ignore_index=True)
    diag_sum["beyond halfway"] = beyond_halfway(df)
    return {
        "protocol sha256": sha256(PROTOCOL), "model": str(MODEL.relative_to(ROOT)),
        "model sha256": sha256(MODEL), "penalties": "not told apart; every attempt scored by the fit",
        "defended end": "the side with the lower GSR mean x over the half defends the left end",
        "halves used": sorted(df.block.unique()), "halves skipped (no cached GSR)": skipped,
        "diagnostics (summed over used halves)": diag_sum,
        "all": summary(df, model.base_rate),
        "all but M1": summary(df[df.match != M1], model.base_rate),
        "M1 (117092)": summary(df[df.match == M1], model.base_rate),
        "per half (test halves left out; they are in the pooled numbers only)": per_half,
        "superseded: sign reading": superseded(pd.concat(old, ignore_index=True), one_end,
                                               model.base_rate),
    }


def report(res: dict) -> None:
    print(f"used {len(res['halves used'])} halves; skipped {res['halves skipped (no cached GSR)']}")
    print("diagnostics", res["diagnostics (summed over used halves)"])
    print("superseded", res["superseded: sign reading"])
    for key in ("all", "all but M1", "M1 (117092)"):
        s = res[key]
        b = s["bootstrap over halves"]
        print(f"\n{key}: n {s['n']}, goals {s['goals']} (rate {s['base rate']:.3f}), "
              f"xG sum {s['xg sum']:.2f}")
        print(f"  AUC {s['auc']:.3f}, 95% over {b['blocks']} halves {b['auc ci95']} "
              f"({b['one-class draws skipped']} one-class draws skipped)")
        print(f"  Brier {s['brier']:.4f} vs base rate {s['brier base rate']:.4f} "
              f"(skill {s['brier skill']:.3f}, 95% {b['brier skill ci95 (extra)']}); "
              f"vs xGHub rate {s['brier xghub base rate']:.4f}")
        print(f"  median distance {s['median distance m']}")
        for r in s["quintile calibration"]:
            print(f"    n={r['n']:3d}  predicted {r['predicted']:.3f}  observed {r['observed']:.3f}")


if __name__ == "__main__":
    result = run()
    OUT.write_text(json.dumps(result, indent=1))
    report(result)
    print(f"\nwrote {OUT}")
