"""The events sweep of protocol v3 (SOCCERTRACK_V2.md, events protocol
E2-E4): the ball input and the event rules chosen in stages on the U-E
windows with the v2 detector's ball, then frozen into events_v3.json (and
its sha256) before any V event.

Players and teams are each window's cached pipeline (output_<w>/events_v1);
only the ball rows are swapped (`score_soccernet.swap_ball_rows`) and the
pipeline is replayed from its teams (`score_soccernet.merged_from_cached`,
`events_on_pitch`). The player rows' hash must be the same for every
configuration and equal to the cached pipeline's.

Stages, each starting from the previous one's choice:

    A   ball input: top-k {1, 4} x selection {select_single_ball,
        ball_link with c_null 1, c_static 2}; the v2 rows are those with
        p_cal >= tau (the calibration's, label-free)
    A2  only if the linker is chosen: c_null {1, 3} x c_static {2, 6}
    B   POSSESSION_RADIUS_M {2, 3, 4, 8} x MIN_POSSESSION_HOLD_S {0.2, 0.4}
    C   CARRY_MIN_TIME_S {0.8, 1.2} x MIN_PASS_DISTANCE_M {0.3, 3}
    D   shots, by count (too few labels for an F1): SHOT_MIN_KMH {10, 30, 50}
        x SHOT_MAX_DIST_FROM_GOAL_M {50, 30} x SHOT_MOUTH_MARGIN_M {None, 5};
        the most permissive setting with at most 3 shots pooled (the most
        shots, then the fewest changes from v1, then the grid order, which
        lists the more permissive values first)
    A'  A again, with B-D fixed (A2 if the linker is newly chosen).

Objective (E3): pass + carry F1 at 1 s pooled over the windows, minus the
pooled chance F1 (`score_events_paired`, 200 draws per window).
Constraint: pass + carry events per window within [0.6, 1.5] x its labels
(if no configuration of a stage meets it, all compete, and this is
recorded). Among those within 0.02 of the best: the fewest parameters
changed from v1, then the fewest events.

Gate G-E (pooled over U-E, with the critique's fix): the frozen
configuration's objective must be at least that of v1 rules on the same
v2 ball (C2: top-4, select_single_ball, v1 rules); otherwise the frozen
configuration is C2, and the record says so. Leave-one-window-out: the
stages are run on one window's objective alone and scored on the other.

    python sweep_events_v2.py --balls W1=V2_1.parquet W2=V2_2.parquet \\
        --calib CKPT.calib.json --out events_v3.json

    # the frozen configuration once on a window (writes events.json and
    # tracks_merged.parquet in output_<w>/<NAME>; prints event counts only)
    python sweep_events_v2.py --apply events_v3.json --window W \\
        [--balls W=PARQUET] [--rules frozen|v1] [--ball-input frozen|v1] \\
        --name NAME

The sweep refuses V and test matches; it runs on U-E (M6, 118578) only, or
on training matches with --allow-train (smoke tests).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import time
from pathlib import Path

import pandas as pd

import score_events_paired as sp
import score_soccernet as sc
from src import events as ev
from src import pixel_scale
from src import soccertrack_v2 as st
from src.paths import DATA_DIR

U_E = ("st2_118578_1st_f030000", "st2_118578_2nd_f015000")
U_MATCH = "118578"
TRAIN = {"118575", "118576", "118577", "128058"}
REFUSED = {"117093", "132877", "128057", "132831"}      # V and test
SRC = "events_v1"
WORK = "events_v2_sweep"

V1_BALL = {"top_k": 4, "select": "v1"}
LINK_DEFAULT = {"c_null": 1.0, "c_static": 2.0}
V1_RULES = {k: getattr(ev, k) for k in ev.EVENT_PARAMS}
BASE = {**V1_BALL, **LINK_DEFAULT, **V1_RULES}
GRID = {
    "A": {"top_k": (1, 4), "select": ("v1", "link")},
    "A2": {"c_null": (1.0, 3.0), "c_static": (2.0, 6.0)},
    "B": {"POSSESSION_RADIUS_M": (2.0, 3.0, 4.0, 8.0),
          "MIN_POSSESSION_HOLD_S": (0.2, 0.4)},
    "C": {"CARRY_MIN_TIME_S": (0.8, 1.2), "MIN_PASS_DISTANCE_M": (0.3, 3.0)},
    "D": {"SHOT_MIN_KMH": (10.0, 30.0, 50.0),
          "SHOT_MAX_DIST_FROM_GOAL_M": (50.0, 30.0),
          "SHOT_MOUTH_MARGIN_M": (None, 5.0)},
}
STAGES = ("A", "A2", "B", "C", "D", "A'")
NEAR_BEST = 0.02
EVENTS_PER_LABEL = (0.6, 1.5)
MAX_SHOTS = 3
ROOT = Path(__file__).resolve().parent
CODE = ("sweep_events_v2.py", "score_events_paired.py", "score_soccernet.py",
        "src/events.py", "src/ball_link.py")


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def match_of(window: str) -> str:
    return json.loads((DATA_DIR / "soccertrack_v2" / f"{window}.json")
                      .read_text())["match"]


def changed(cfg: dict) -> dict:
    """The configuration's departures from v1 (c_null / c_static count
    only with the linker)."""
    out = {k: v for k, v in cfg.items() if v != BASE[k]}
    if cfg["select"] != "link":
        out = {k: v for k, v in out.items() if k not in LINK_DEFAULT}
    return out


def key_of(cfg: dict) -> str:
    return json.dumps(changed(cfg), sort_keys=True)


def grid(stage: str, cfg: dict) -> list[dict]:
    """The stage's configurations around ``cfg``, in grid order."""
    out = [{}]
    for name, values in GRID["A" if stage == "A'" else stage].items():
        out = [{**o, name: v} for o in out for v in values]
    return [{**cfg, **o} for o in out]


class Sweep:
    """Events of every configuration on every window, computed once."""

    def __init__(self, dirs: dict[str, Path], tau: float | None,
                 n_chance: int = sp.N_CHANCE, src_hash: dict | None = None):
        self.dirs, self.tau, self.n_chance = dirs, tau, n_chance
        self.cal = {w: st.calibration(match_of(w)) for w in dirs}
        self.truths = {w: sp.labels_of(w) for w in dirs}
        self.durations = {w: sp.duration_of(w, d) for w, d in dirs.items()}
        self.labels = {w: sum(len(t) for t in self.truths[w].values())
                       for w in dirs}
        self.src_hash = src_hash or {}
        self.hashes: dict[str, set] = {w: set() for w in dirs}
        self._metric, self._events, self._scores = {}, {}, {}

    def metric(self, w: str, cfg: dict) -> pd.DataFrame:
        link = cfg["select"] == "link"
        key = (w, cfg["top_k"], cfg["select"],
               (cfg["c_null"], cfg["c_static"]) if link else None)
        if key not in self._metric:
            with contextlib.redirect_stdout(io.StringIO()):
                merged = sc.merged_from_cached(
                    self.dirs[w], self.cal[w], select=cfg["select"],
                    link_params=({k: cfg[k] for k in LINK_DEFAULT}
                                 if link else None),
                    top_k=cfg["top_k"], tau=self.tau)
            h = merged.attrs["player_hash"]
            self.hashes[w].add(h)
            if len(self.hashes[w]) > 1 or (
                    w in self.src_hash and h != self.src_hash[w]):
                raise RuntimeError(f"{w}: player rows differ across ball "
                                   "configurations")
            self._metric[key] = pixel_scale.to_pitch_metres(
                merged, lambda uv, cal=self.cal[w]: st.image_to_pitch(uv, cal))
        return self._metric[key]

    def events(self, w: str, cfg: dict) -> list[dict]:
        key = (w, key_of(cfg))
        if key not in self._events:
            rules = {k: v for k, v in changed(cfg).items() if k in V1_RULES}
            with contextlib.redirect_stdout(io.StringIO()):
                self._events[key] = ev.detect_events(
                    self.metric(w, cfg), absolute_pitch=True,
                    ball_time_aware=True, params=rules)
        return self._events[key]

    def shots(self, cfg: dict, scope) -> int:
        return sum(e["event_type"] == "shot" for w in scope
                   for e in self.events(w, cfg))

    def score(self, cfg: dict, scope, tols=(sp.OBJECTIVE_TOL_S,)) -> dict:
        """Objective and constraint of ``cfg`` over the windows ``scope``."""
        key = (key_of(cfg), tuple(scope), tuple(tols))
        if key not in self._scores:
            evs = {w: self.events(w, cfg) for w in scope}
            s = sp.score(evs, {w: self.truths[w] for w in scope},
                         {w: self.durations[w] for w in scope}, tols=tols,
                         n_chance=self.n_chance)
            n = {w: s[f"{sp.OBJECTIVE_TOL_S:g}s"][w]["events"] for w in scope}
            s["meets"] = all(EVENTS_PER_LABEL[0] * self.labels[w] <= n[w]
                             <= EVENTS_PER_LABEL[1] * self.labels[w]
                             for w in scope)
            s["pass_carry_events"] = n
            s["shots"] = {w: self.shots(cfg, [w]) for w in scope}
            self._scores[key] = s
        return self._scores[key]

    def choose(self, stage: str, cfgs: list[dict], scope, log: list) -> dict:
        """E3's choice among ``cfgs`` (D: the shot rule), logged."""
        rows = []
        for i, cfg in enumerate(cfgs):
            if stage == "D":
                rows.append({"config": changed(cfg), "i": i, "cfg": cfg,
                             "shots": self.shots(cfg, scope)})
                continue
            s = self.score(cfg, scope)
            rows.append({"config": changed(cfg), "i": i, "cfg": cfg,
                         "objective": s["objective"]["pooled"],
                         "per_window": {w: s["objective"][w] for w in scope},
                         "events": s["pass_carry_events"],
                         "labels": {w: self.labels[w] for w in scope},
                         "meets": s["meets"], "shots": s["shots"]})
        if stage == "D":
            ok = [r for r in rows if r["shots"] <= MAX_SHOTS]
            pick = (max(ok, key=lambda r: (r["shots"], -len(r["config"]),
                                           -r["i"])) if ok else
                    min(rows, key=lambda r: (r["shots"], len(r["config"]),
                                             r["i"])))
            note = "" if ok else f"no setting gives <= {MAX_SHOTS} shots"
        else:
            ok = [r for r in rows if r["meets"]]
            pool = ok or rows
            best = max(r["objective"] for r in pool)
            near = [r for r in pool if r["objective"] >= best - NEAR_BEST]
            pick = min(near, key=lambda r: (len(r["config"]),
                                            sum(r["events"].values()), r["i"]))
            note = "" if ok else "no configuration meets the event-count constraint"
        log.append({"stage": stage, "scope": list(scope),
                    "rows": [{k: v for k, v in r.items() if k not in ("i", "cfg")}
                             for r in rows],
                    "chosen": changed(pick["cfg"]), "note": note})
        return pick["cfg"]

    def staged(self, scope, stages=STAGES) -> tuple[dict, list]:
        """The staged choice on the windows ``scope``: (config, log)."""
        cfg, log = dict(BASE), []
        if "A" in stages:
            cfg = self.choose("A", grid("A", cfg), scope, log)
        if "A2" in stages and cfg["select"] == "link":
            cfg = self.choose("A2", grid("A2", cfg), scope, log)
        for stage in ("B", "C", "D"):
            if stage in stages:
                cfg = self.choose(stage, grid(stage, cfg), scope, log)
        if "A'" in stages:
            was_link = cfg["select"] == "link"
            cfg = self.choose("A'", grid("A'", cfg), scope, log)
            if "A2" in stages and cfg["select"] == "link" and not was_link \
                    and not any(r["stage"] == "A2" for r in log):
                cfg = self.choose("A2", grid("A2", cfg), scope, log)
        return cfg, log


def prepare(window: str, balls: str | None, allow_train: bool) -> Path:
    """output_<w>/WORK: the cached pipeline with the v2 ball rows (reused
    while the rows file is unchanged, so the grass fractions are measured
    once), or the source run itself without --balls."""
    m = match_of(window)
    if m in REFUSED:
        raise SystemExit(f"{window}: match {m} is V or test; the sweep never "
                         "runs there")
    if m != U_MATCH and not (allow_train and m in TRAIN):
        raise SystemExit(f"{window}: the sweep runs on U-E ({U_MATCH}) only "
                         "(--allow-train for a training-match smoke test)")
    src = Path(f"output_{window}") / SRC
    if not (src / "tracks_teams.parquet").exists():
        raise SystemExit(f"{src}: no cached pipeline; run it first "
                         "(eval_soccertrack_v2.py --run --calibrated)")
    if balls is None:
        return src
    dst = Path(f"output_{window}") / WORK
    marker = {"src": str(src), "balls": str(balls),
              "sha256": sha256_of(balls)}
    done = dst / "balls.json"
    if not (done.exists() and json.loads(done.read_text()) == marker
            and (dst / "tracks_grass.parquet").exists()):
        sc.swap_ball_rows(src, dst, pd.read_parquet(balls))
        done.write_text(json.dumps(marker))
    return dst


def tau_of(args) -> float | None:
    if args.tau is not None:
        return args.tau
    if args.calib:
        return float(json.loads(Path(args.calib).read_text())["tau"])
    return None


def sweep(args):
    windows = args.windows or list(U_E)
    balls = dict(b.split("=", 1) for b in args.balls or [])
    unknown = set(balls) - set(windows)
    if unknown:
        raise SystemExit(f"--balls for windows not swept: {sorted(unknown)}")
    if balls and set(balls) != set(windows):
        raise SystemExit("--balls must name every window, or none")
    tau = tau_of(args)
    if balls and tau is None:
        raise SystemExit("v2 rows need --calib or --tau")
    stages = tuple(args.stages.split(",")) if args.stages else STAGES
    t0 = time.time()
    dirs = {w: prepare(w, balls.get(w), args.allow_train) for w in windows}
    # The cached pipeline's players, to which every configuration's must
    # be identical.
    src_hash = {}
    for w in windows:
        with contextlib.redirect_stdout(io.StringIO()):
            src_hash[w] = sc.merged_from_cached(
                Path(f"output_{w}") / SRC,
                st.calibration(match_of(w))).attrs["player_hash"]
    s = Sweep(dirs, tau if balls else None, args.chance, src_hash)
    print(f"windows {windows}, tau {s.tau}, labels {s.labels}", flush=True)

    frozen, log = s.staged(windows, stages)
    c2 = dict(BASE)
    full = s.score(frozen, windows, tols=sp.TOLERANCES_S)
    base = s.score(c2, windows, tols=sp.TOLERANCES_S)
    gate = {"frozen": full["objective"]["pooled"],
            "c2": base["objective"]["pooled"],
            "per_window": {w: {"frozen": full["objective"][w],
                               "c2": base["objective"][w]} for w in windows}}
    gate["passed"] = gate["frozen"] >= gate["c2"]
    chosen = frozen if gate["passed"] else c2
    loo = []
    if len(windows) > 1:
        for w in windows:
            rest = [x for x in windows if x != w]
            cfg, _ = s.staged(rest, stages)
            loo.append({"chosen_on": rest, "scored_on": w,
                        "config": changed(cfg),
                        "objective": s.score(cfg, [w])["objective"][w],
                        "c2": s.score(c2, [w])["objective"][w]})

    record = {
        "protocol": "events v3 (SOCCERTRACK_V2.md; design_v2 events_protocol "
                    "E2-E4, critique: G-E pooled with leave-one-window-out)",
        "windows": windows, "source_run": SRC, "stages": list(stages),
        "tau": s.tau, "calib": args.calib,
        "balls": {w: {"path": str(p), "sha256": sha256_of(p)}
                  for w, p in balls.items()},
        "v1": BASE, "frozen": chosen, "frozen_changes": changed(chosen),
        "swept_choice": changed(frozen), "c2": c2,
        "gate": gate, "leave_one_window_out": loo,
        "score_frozen": full, "score_c2": base, "log": log,
        "player_hash": {w: sorted(h) for w, h in s.hashes.items()},
        "code_sha256": {f: sha256_of(ROOT / f) for f in CODE},
        "elapsed_s": round(time.time() - t0),
    }
    out = Path(args.out)
    out.write_text(json.dumps(record, indent=1, default=str))
    digest = sha256_of(out)
    Path(f"{out}.sha256").write_text(f"{digest}  {out.name}\n")
    for r in log:
        print(f"[{r['stage']}] chosen {r['chosen']} {r['note']}")
    print(f"gate G-E {'passed' if gate['passed'] else 'FAILED -> C2'}: "
          f"frozen {gate['frozen']:+.3f} vs C2 {gate['c2']:+.3f}")
    for r in loo:
        print(f"LOO: chosen on {r['chosen_on']} -> {r['config']}; on "
              f"{r['scored_on']}: {r['objective']:+.3f} vs C2 {r['c2']:+.3f}")
    print(f"wrote {out} (sha256 {digest}) [{time.time() - t0:.0f} s]")


def apply(args):
    """The frozen (or C2) configuration on one window, written as a run."""
    path = Path(args.apply)
    digest = Path(f"{path}.sha256").read_text().split()[0]
    if sha256_of(path) != digest:
        raise SystemExit(f"{path} does not match its sha256 file")
    record = json.loads(path.read_text())
    w = args.window
    m = match_of(w)
    if m in {"128057", "132831"}:
        raise SystemExit(f"{w}: test match")
    src = Path(f"output_{w}") / SRC
    dst = sp.run_dir_of(args.name, w)
    if (dst / "applied.json").exists() and not args.overwrite:
        raise SystemExit(f"{dst} was applied already (one run per window; "
                         "--overwrite to redo it)")
    balls = dict(b.split("=", 1) for b in args.balls or [])
    rows_path = balls.get(w)
    rows = pd.read_parquet(rows_path) if rows_path else None
    sc.swap_ball_rows(src, dst, rows)
    frozen = {**BASE, **record["frozen"]}
    ball_cfg = frozen if args.ball_input == "frozen" else BASE
    rule_cfg = frozen if args.rules == "frozen" else BASE
    if rows is None:
        # The source's own (COCO) ball through the v1 path: tau and top-k
        # belong to the v2 detector's rows.
        select, link_params, top_k, tau = "v1", None, None, None
    else:
        select, top_k, tau = ball_cfg["select"], ball_cfg["top_k"], record["tau"]
        link_params = ({k: ball_cfg[k] for k in LINK_DEFAULT}
                       if select == "link" else None)
    rules = {k: v for k, v in changed(rule_cfg).items() if k in V1_RULES}
    with contextlib.redirect_stdout(io.StringIO()):
        events, merged = sc.events_from_cached(
            dst, st.calibration(m), select=select, link_params=link_params,
            top_k=top_k, tau=tau, event_params=rules)
    sc.write_events(dst / "events.json", events)
    merged.to_parquet(dst / "tracks_merged.parquet")
    applied = {"events_v3": str(path), "sha256": digest, "window": w,
               "balls": ({"path": rows_path, "sha256": sha256_of(rows_path)}
                         if rows_path else "source run's"),
               "select": select, "link_params": link_params, "top_k": top_k,
               "tau": tau, "rules": rules,
               "player_hash": merged.attrs["player_hash"]}
    (dst / "applied.json").write_text(json.dumps(applied, indent=1))
    counts = pd.Series([e["event_type"] for e in events]).value_counts()
    print(f"{dst}: {len(events)} events {counts.to_dict()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--windows", nargs="+", help=f"default {list(U_E)}")
    ap.add_argument("--balls", nargs="+", metavar="W=PARQUET",
                    help="dense v2 ball rows per window "
                         "(detect_ball_heatmap.py --v2); without them, the "
                         "source run's own ball rows")
    ap.add_argument("--calib", help="CKPT.calib.json, for tau")
    ap.add_argument("--tau", type=float, help="instead of --calib")
    ap.add_argument("--stages", help=f"comma-separated subset of {STAGES}")
    ap.add_argument("--chance", type=int, default=sp.N_CHANCE,
                    help="chance draws per window")
    ap.add_argument("--allow-train", action="store_true",
                    help="allow training-match windows (smoke tests)")
    ap.add_argument("--out", help="events_v3.json")
    ap.add_argument("--apply", help="events_v3.json to apply to --window")
    ap.add_argument("--window")
    ap.add_argument("--name", help="run directory under output_<window>/ "
                                   "(or a template with {w})")
    ap.add_argument("--rules", choices=("frozen", "v1"), default="frozen")
    ap.add_argument("--ball-input", choices=("frozen", "v1"),
                    default="frozen")
    ap.add_argument("--overwrite", action="store_true",
                    help="redo an --apply run directory")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        _check()
    elif args.apply:
        if not (args.window and args.name):
            ap.error("--apply needs --window and --name")
        apply(args)
    else:
        if not args.out:
            ap.error("--out is required")
        sweep(args)


def _check():
    """Synthetic: grid sizes and order, departures from v1, and the
    choice rules (constraint, 0.02 band, fewest changes, shot rule) on a
    stub scorer."""
    assert [len(grid(s, BASE)) for s in ("A", "A2", "B", "C", "D", "A'")] \
        == [4, 4, 8, 4, 12, 4]
    assert changed(BASE) == {} and grid("D", BASE)[0] == BASE
    assert changed({**BASE, "c_null": 3.0}) == {}
    assert changed({**BASE, "select": "link", "c_null": 3.0}) == \
        {"select": "link", "c_null": 3.0}

    class Stub(Sweep):
        def __init__(self, table, shots):
            self.table, self.n_shots, self.labels = table, shots, {"w": 10}

        def score(self, cfg, scope, tols=None):
            obj, n = self.table(cfg)
            return {"objective": {"pooled": obj, "w": obj},
                    "pass_carry_events": {"w": n},
                    "meets": 6 <= n <= 15, "shots": {"w": 0}}

        def shots(self, cfg, scope):
            return self.n_shots(cfg)

    def table(cfg):
        # The linker is best by 0.01 (inside the band: v1 kept); radius 3
        # is best by 0.05 but radius 2 overfires (constraint).
        obj = 0.10 + 0.01 * (cfg["select"] == "link")
        obj += {2.0: 0.2, 3.0: 0.05}.get(cfg["POSSESSION_RADIUS_M"], 0.0)
        n = 30 if cfg["POSSESSION_RADIUS_M"] == 2.0 else 12
        return obj, n - (cfg["top_k"] == 1)

    def shots(cfg):
        return 9 if cfg["SHOT_MIN_KMH"] == 10 else (
            3 if cfg["SHOT_MOUTH_MARGIN_M"] == 5.0 else 5)

    cfg, log = Stub(table, shots).staged(["w"])
    assert cfg["select"] == "v1" and cfg["top_k"] == 4, cfg
    assert cfg["POSSESSION_RADIUS_M"] == 3.0
    assert cfg["SHOT_MIN_KMH"] == 30.0 and cfg["SHOT_MOUTH_MARGIN_M"] == 5.0
    assert cfg["SHOT_MAX_DIST_FROM_GOAL_M"] == 50.0
    assert [r["stage"] for r in log] == ["A", "B", "C", "D", "A'"]
    print("sweep_events_v2 check ok")


if __name__ == "__main__":
    main()
