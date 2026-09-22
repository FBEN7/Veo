#!/usr/bin/env python3
"""Score detected events against SoccerNet ball-action labels.

Twelve hand-annotated intervals were the only ground truth this project had
for events, which is too few to distinguish a working detector from a lucky
one. SoccerNet's ball action spotting set labels every Pass, Drive, Header,
Cross and Shot with a timestamp, so a 90-second window carries 60 of them.

The chance control is not optional here. These labels are dense -- one every
1.5 seconds -- so a detector emitting roughly the right *number* of events at
arbitrary times matches a large share of them by coincidence. The first run
scored F1 0.88 at +/-5 s tolerance, which looked like success until randomly
placed timestamps scored 0.91. Every figure is therefore reported against the
null of the same event count drawn uniformly over the clip, with an exact
permutation p-value.

Usage:
    python score_soccernet.py --clip CLIP.mp4 --labels Labels-ball.json \
        --offset 1820 --out output_soccernet

SoccerNet data is under a non-commercial NDA. It is a measuring instrument:
nothing derived from it belongs in the product or in this repository.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# SoccerNet ball actions -> the event types this pipeline emits.
LABEL_MAP = {
    "PASS": "pass",
    "HIGH PASS": "pass",
    "CROSS": "pass",
    "HEADER": "pass",
    "DRIVE": "carry",
    "BALL PLAYER BLOCK": "recovery",
    "PLAYER SUCCESSFUL TACKLE": "recovery",
}

OURS_TO_GROUP = {
    "pass": "pass",
    "carry": "carry",
    "recovery": "recovery",
    "tackle": "recovery",
    "interception": "recovery",
}

TOLERANCES = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)


def load_window(labels_path: str, offset_s: float, duration_s: float):
    """Labels falling inside the clip, with times relative to its first frame."""
    data = json.loads(Path(labels_path).read_text())
    out = []
    for ann in data.get("annotations", []):
        t = float(ann["position"]) / 1000.0 - offset_s
        if not (0.0 <= t <= duration_s):
            continue
        out.append(dict(t=t, label=ann["label"],
                        group=LABEL_MAP.get(ann["label"])))
    return sorted(out, key=lambda a: a["t"])


def match_one_to_one(preds, truths, tolerance):
    """Greedy nearest-first matching, each side used once.

    Nearest-first rather than in time order: a detection should be credited to
    the label it is closest to, not to whichever one happens to come first.
    """
    pairs = sorted(
        ((abs(p - t), pi, ti)
         for pi, p in enumerate(preds) for ti, t in enumerate(truths)
         if abs(p - t) <= tolerance),
        key=lambda x: x[0],
    )
    used_p, used_t, matched = set(), set(), []
    for dt, pi, ti in pairs:
        if pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matched.append((pi, ti, dt))
    fp = [i for i in range(len(preds)) if i not in used_p]
    fn = [i for i in range(len(truths)) if i not in used_t]
    return matched, fp, fn


def chance_control(events, labels, tolerance, duration_s,
                   n_draws=2000, seed=1):
    """What the same number of events, placed at random, would score."""
    rng = np.random.default_rng(seed)
    out = {}
    for group in ("pass", "carry", "recovery"):
        truths = sorted(l["t"] for l in labels if l["group"] == group)
        preds = sorted(e["timestamp_s"] for e in events
                       if OURS_TO_GROUP.get(e["event_type"]) == group)
        if not preds or not truths:
            out[group] = dict(f1=0.0, chance=0.0, p=float("nan"))
            continue

        pairs, _, _ = match_one_to_one(preds, truths, tolerance)
        p = len(pairs) / len(preds)
        r = len(pairs) / len(truths)
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0

        draws = np.empty(n_draws)
        for k in range(n_draws):
            rand = sorted(rng.uniform(0.0, duration_s, len(preds)))
            pr, _, _ = match_one_to_one(rand, truths, tolerance)
            pp = len(pr) / len(preds)
            rr = len(pr) / len(truths)
            draws[k] = 2 * pp * rr / (pp + rr) if (pp + rr) else 0.0

        out[group] = dict(f1=f1, chance=float(draws.mean()),
                          p=float((draws >= f1).mean()))
    return out


def probe_clip(path: str) -> dict:
    """Identify the clip, or refuse to run.

    Every stage of this pipeline is cached, which makes a mismatch between
    cache and input silent rather than loud. The failure that prompted this:
    passing --clip /dev/null to reuse cached detections left every cached
    stage valid, but camera motion validation could not read the video, so it
    reported the estimate as unusable and the run continued without
    compensation. The output was plausible -- carry detection simply looked
    weak again -- and only contradicted a result measured minutes earlier.

    A measurement tool that quietly downgrades is worse than one that breaks:
    the numbers still look like numbers.
    """
    import cv2

    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"clip not found: {path}")

    cap = cv2.VideoCapture(str(p))
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise SystemExit(
            f"cannot decode video frames from {path}. "
            "Cached stages would still load and the run would produce "
            "believable but wrong numbers, so this is fatal rather than a "
            "warning.")
    info = dict(
        path=str(p.resolve()),
        size_bytes=p.stat().st_size,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=round(float(cap.get(cv2.CAP_PROP_FPS)) or 25.0, 3),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def check_cache_provenance(out_dir: Path, clip_info: dict) -> None:
    """Refuse to mix cached stages with a different clip.

    Output directories get reused -- and during development a camera motion
    estimate was once copied between them by hand. Nothing downstream would
    notice: the arrays have the right shape and the numbers come out looking
    ordinary.
    """
    manifest = out_dir / "clip.json"
    cached_stages = [p.name for p in (
        out_dir / "tracks.parquet", out_dir / "tracks_teams.parquet",
        out_dir / "tracks_grass.parquet", out_dir / "camera_motion.npy",
        out_dir / "profile.json") if p.exists()]

    if not manifest.exists() and cached_stages:
        # A directory holding cached stages but no record of what produced
        # them cannot be verified. Writing the manifest here would adopt
        # whatever clip happened to be passed -- which is exactly the mistake
        # this function exists to catch, and it fired on the first attempt to
        # test it.
        raise SystemExit(
            f"{out_dir} holds cached stages ({', '.join(cached_stages)}) but "
            "no clip.json recording which clip built them, so they cannot be "
            "checked against the clip given.\n"
            "Delete the directory to rebuild, or use a different --out.")

    if manifest.exists():
        previous = json.loads(manifest.read_text())
        differing = {k: (previous.get(k), clip_info.get(k))
                     for k in ("size_bytes", "width", "height", "n_frames")
                     if previous.get(k) != clip_info.get(k)}
        if differing:
            detail = ", ".join(f"{k}: cached {a} vs given {b}"
                               for k, (a, b) in differing.items())
            raise SystemExit(
                f"{out_dir} holds cached stages built from a different clip "
                f"({detail}).\n"
                f"  cached: {previous.get('path')}\n"
                f"  given : {clip_info['path']}\n"
                "Use a different --out, or delete the directory to rebuild.")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(clip_info, indent=2))


def run_pipeline(clip: str, out_dir: Path, return_tracks: bool = False,
                 team_override: dict[int, str] | None = None):
    """Detect, track and emit events for the clip, caching each stage.

    ``return_tracks`` also hands back the metric tracks the events were
    derived from, so a diagnostic can look at the possession timeline that
    produced a given event rather than reconstructing the pipeline and
    drifting out of step with it.

    ``team_override`` replaces the team of each track after the cached stages
    are loaded, so an alternative team assignment can be measured end to end
    without re-running detection. Nothing between detection and events reads
    the team column, so substituting it here is equivalent to having assigned
    teams differently in the first place. Tracks absent from the mapping keep
    no team, which is how an assignment declining to place a track is
    expressed.
    """
    from src.detect_track_hybrid import run as run_detection
    from src.team_assignment_v2 import assign_teams_v2 as assign_teams
    from src import (auto_tune, pixel_scale, ball_selection, track_reid,
                     ball_pitch_filter, player_filter, camera_motion,
                     ground_plane)
    from src import events as ev_module

    clip_info = probe_clip(clip)
    check_cache_provenance(out_dir, clip_info)
    print(f"clip: {clip_info['width']}x{clip_info['height']} @ "
          f"{clip_info['fps']:.0f} fps, {clip_info['n_frames']} frames "
          f"({clip_info['size_bytes'] / 1e6:.1f} MB)")

    profile_path = out_dir / "profile.json"
    if profile_path.exists():
        profile = auto_tune.VideoProfile.load(profile_path)
        if profile.n_frames != clip_info["n_frames"]:
            raise SystemExit(
                f"cached profile describes {profile.n_frames} frames, the "
                f"clip has {clip_info['n_frames']}. Delete {profile_path} "
                "to re-derive.")
        print(f"cached profile:\n{profile.summary()}")
    else:
        print("deriving parameters from the clip...")
        profile = auto_tune.profile_video(clip)
        profile.save(profile_path)

    raw = out_dir / "tracks.parquet"
    if raw.exists():
        tracks = pd.read_parquet(raw)
    else:
        print("detecting...")
        tracks = run_detection(
            clip, stride=1, model_name="yolov8m.pt",
            conf_player=profile.conf_player, conf_ball=profile.conf_ball,
            out_path=str(raw), max_seconds=0, imgsz=profile.imgsz,
            ball_detection_method="yolo")
        tracks.to_parquet(raw)

    teamed = out_dir / "tracks_teams.parquet"
    if teamed.exists():
        tracks = pd.read_parquet(teamed)
    else:
        print("assigning teams...")
        tracks = assign_teams(clip, tracks)
        tracks.to_parquet(teamed)

    grass = out_dir / "tracks_grass.parquet"
    if grass.exists():
        tracks = pd.read_parquet(grass)
    else:
        print("annotating grass fraction...")
        tracks = ball_pitch_filter.annotate_grass_fraction(tracks, clip)
        tracks.to_parquet(grass)

    if team_override is not None:
        tracks = tracks.copy()
        is_player = tracks.cls == "player"
        tracks.loc[is_player, "team"] = (
            tracks.loc[is_player, "track_id"].map(team_override))

    filtered = ball_pitch_filter.filter_ball_by_pitch(tracks, verbose=True)
    scale = pixel_scale.estimate_px_per_m(filtered)

    # Camera motion: a pan is otherwise indistinguishable from every player
    # sprinting sideways, and carry detection asks whether a player and the
    # ball move together.
    if profile.compensate_camera:
        motion_path = out_dir / "camera_motion.npy"
        if motion_path.exists():
            motion = np.load(motion_path)
            # A motion array of the wrong length is the signature of one
            # copied from another clip, which nothing downstream would notice:
            # compensate() clips the index and returns ordinary-looking
            # positions.
            if abs(len(motion) - clip_info["n_frames"]) > 2:
                raise SystemExit(
                    f"{motion_path} has {len(motion)} entries for a "
                    f"{clip_info['n_frames']}-frame clip. Delete it to "
                    "re-estimate.")
        else:
            print("estimating camera motion...")
            motion = camera_motion.estimate_camera_motion(clip)
            np.save(motion_path, motion)

        usable, why = camera_motion.validate_motion(clip, motion)
        if usable:
            filtered = camera_motion.compensate(filtered, motion)
            print(f"camera: compensated ({why})")
        elif "could not measure" in why or "no motion estimated" in why:
            # The validator could not run, as distinct from running and
            # rejecting the estimate. Carrying on uncompensated here is what
            # produced a full set of wrong numbers once already: carry
            # detection depends on compensation, so the run would report it as
            # weak rather than as unmeasured.
            raise SystemExit(
                f"camera motion could not be validated ({why}). "
                "Refusing to continue uncompensated: carry detection depends "
                "on compensation, so the run would report weak results rather "
                "than absent ones.")
        else:
            # A genuine rejection is a real finding, not an error.
            print(f"camera: NOT compensated - validator rejected the "
                  f"estimate - {why}")
    else:
        print("camera: static, not compensated")

    capped = player_filter.filter_players(filtered, verbose=True)
    selected = ball_selection.select_single_ball(
        capped, scale, fps=profile.fps, verbose=True)
    merged = track_reid.merge_fragments(
        selected, scale, fps=profile.fps, verbose=True)

    # A scale taken from player height is the scale for motion across the
    # view, not into it, so it overstates pixels per metre and every distance
    # comes out short. The ground plane measures by how much. It does not
    # replace the coordinate system -- see ground_plane.effective_px_per_m for
    # why the geometrically correct map measures worse than this scalar.
    plane = ground_plane.load_or_build(
        clip, merged, clip_info["width"], clip_info["height"],
        out_dir / "ground_plane.json")
    calibrated = (ground_plane.effective_px_per_m(merged, plane, verbose=True)
                  if plane is not None else None)

    metric, absolute = pixel_scale.prepare_tracks_for_events(
        merged, None, verbose=True, px_per_m=calibrated)
    events = ev_module.detect_events(metric, absolute_pitch=absolute)

    diag = ball_selection.ball_track_diagnostics(merged, scale, fps=profile.fps)
    print(f"\nball coverage {diag['coverage_pct']:.1f}%, "
          f"{len(events)} events, scale {scale:.1f} px/m")
    if return_tracks:
        return events, profile, metric
    return events, profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--offset", type=float, required=True,
                    help="match time, in seconds, of the clip's first frame")
    ap.add_argument("--out", default="output_soccernet")
    args = ap.parse_args()

    out_dir = Path(args.out)
    events, profile = run_pipeline(args.clip, out_dir)
    duration = profile.n_frames / profile.fps

    labels = load_window(args.labels, args.offset, duration)
    scored = [l for l in labels if l["group"]]
    skipped = [l for l in labels if not l["group"]]

    # An offset that misses the clip yields an empty window, and every
    # precision and recall below would then be 0.00 -- which reads as a
    # detector that found nothing rather than as a window containing nothing.
    if not scored:
        raise SystemExit(
            f"no scorable labels between {args.offset:.0f} and "
            f"{args.offset + duration:.0f} s of the match. Check --offset: it "
            "is the match time of the clip's FIRST FRAME, and getting it "
            "wrong misaligns every label against every detection.")
    if len(scored) < 10:
        print(f"\n!! only {len(scored)} scorable labels in this window. "
              "Precision and recall will be dominated by single events, and "
              "the permutation test has little power.")

    print("\n" + "=" * 74)
    print(f"EVENTS vs SOCCERNET BALL ACTIONS  ({duration:.0f} s clip, "
          f"match time {args.offset:.0f}-{args.offset + duration:.0f} s)")
    print("=" * 74)
    print(f"labels in window : {len(labels)}  "
          f"{dict(Counter(l['label'] for l in labels))}")
    print(f"  scored         : {len(scored)}")
    print(f"  not attempted  : {len(skipped)}  "
          f"{dict(Counter(l['label'] for l in skipped))}")
    print(f"our events       : {len(events)}  "
          f"{dict(Counter(e['event_type'] for e in events))}")

    results = {}
    for tol in TOLERANCES:
        print(f"\n--- tolerance +/-{tol:g} s "
              + "-" * 46)
        print(f"  {'group':<10} {'truth':>6} {'ours':>6} {'TP':>5} "
              f"{'precision':>10} {'recall':>8} {'F1':>7} {'median dt':>10}")
        agg = defaultdict(lambda: dict(tp=0, pred=0, gt=0, dts=[]))
        for group in ("pass", "carry", "recovery"):
            truths = sorted(l["t"] for l in scored if l["group"] == group)
            preds = sorted(e["timestamp_s"] for e in events
                           if OURS_TO_GROUP.get(e["event_type"]) == group)
            pairs, fp, fn = match_one_to_one(preds, truths, tol)
            prec = len(pairs) / len(preds) if preds else 0.0
            rec = len(pairs) / len(truths) if truths else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            dts = [d for _, _, d in pairs]
            print(f"  {group:<10} {len(truths):>6} {len(preds):>6} "
                  f"{len(pairs):>5} {prec:>10.2f} {rec:>8.2f} {f1:>7.2f} "
                  f"{(np.median(dts) if dts else float('nan')):>10.2f}")
            agg["any"]["tp"] += len(pairs)
            agg["any"]["pred"] += len(preds)
            agg["any"]["gt"] += len(truths)
            agg["any"]["dts"] += dts

        a = agg["any"]
        prec = a["tp"] / a["pred"] if a["pred"] else 0.0
        rec = a["tp"] / a["gt"] if a["gt"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {'any':<10} {a['gt']:>6} {a['pred']:>6} {a['tp']:>5} "
              f"{prec:>10.2f} {rec:>8.2f} {f1:>7.2f} "
              f"{(np.median(a['dts']) if a['dts'] else float('nan')):>10.2f}")
        results[f"tol_{tol}"] = dict(precision=prec, recall=rec, f1=f1)

    print("\n" + "=" * 74)
    print("AGAINST CHANCE  (same event count, uniformly random times)")
    print("=" * 74)
    print("These labels are dense -- one every 1.5 s -- so coincidental")
    print("matches are common and the raw scores above overstate the result.")
    print(f"\n  {'group':<10} {'tol':>6} {'ours F1':>8} {'chance':>8} {'p':>8}")
    for tol in (0.25, 0.5, 1.0, 2.0):
        for group in ("pass", "carry"):
            c = chance_control(events, scored, tol, duration)[group]
            mark = "" if np.isnan(c["p"]) else (
                "  <- above chance" if c["p"] < 0.05 else "")
            print(f"  {group:<10} {tol:>6.2f} {c['f1']:>8.2f} "
                  f"{c['chance']:>8.2f} {c['p']:>8.3f}{mark}")
            results[f"chance_{group}_{tol}"] = c

    (out_dir / "score.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {out_dir / 'score.json'}")


if __name__ == "__main__":
    main()
