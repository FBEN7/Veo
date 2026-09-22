#!/usr/bin/env python3
"""Run the pipeline on footage with no ground truth, and report its behaviour.

Every accuracy figure in this project comes from SoccerNet or Roboflow, both
720p or 1080p broadcast. The footage this product is actually for is a Veo
camera at 640x360, and nothing has been measured on it.

Without labels there is no precision or recall to report, so this reports what
can be checked against what football is: roughly 22 players on the pitch, one
ball, speeds a human can reach, distances a player can cover. Those are weak
tests -- they catch a pipeline that is badly wrong, not one that is subtly so,
and the project has already been misled once by exactly this kind of
plausibility check. They are reported as observations, never as accuracy.

Usage:
    python run_veo_analysis.py --clip veo_clip.mp4 --out output_veo
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import score_soccernet as sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="output_veo")
    args = ap.parse_args()

    from src import (auto_tune, pixel_scale, ball_selection, track_reid,
                     ball_pitch_filter, player_filter, camera_motion, stats)
    from src import events as ev_module
    from src.detect_track_hybrid import run as run_detection
    from src.team_assignment_v2 import assign_teams_v2 as assign_teams

    out_dir = Path(args.out)
    clip_info = sc.probe_clip(args.clip)
    sc.check_cache_provenance(out_dir, clip_info)

    print("=" * 74)
    print(f"VEO FOOTAGE  {Path(args.clip).name}")
    print("=" * 74)
    print(f"clip           : {clip_info['width']}x{clip_info['height']} @ "
          f"{clip_info['fps']:.0f} fps, {clip_info['n_frames']} frames "
          f"({clip_info['n_frames'] / clip_info['fps']:.0f} s)")

    profile_path = out_dir / "profile.json"
    if profile_path.exists():
        profile = auto_tune.VideoProfile.load(profile_path)
    else:
        print("\nderiving parameters from this footage...")
        profile = auto_tune.profile_video(args.clip)
        profile.save(profile_path)
    print(f"\nDERIVED PARAMETERS\n{profile.summary()}")

    raw = out_dir / "tracks.parquet"
    if raw.exists():
        tracks = pd.read_parquet(raw)
    else:
        print("\ndetecting...")
        tracks = run_detection(
            args.clip, stride=1, model_name="yolov8m.pt",
            conf_player=profile.conf_player, conf_ball=profile.conf_ball,
            out_path=str(raw), max_seconds=0, imgsz=profile.imgsz,
            ball_detection_method="yolo")
        tracks.to_parquet(raw)

    teamed = out_dir / "tracks_teams.parquet"
    if teamed.exists():
        tracks = pd.read_parquet(teamed)
    else:
        print("assigning teams...")
        tracks = assign_teams(args.clip, tracks)
        tracks.to_parquet(teamed)

    grass = out_dir / "tracks_grass.parquet"
    if grass.exists():
        tracks = pd.read_parquet(grass)
    else:
        print("annotating grass fraction...")
        tracks = ball_pitch_filter.annotate_grass_fraction(tracks, args.clip)
        tracks.to_parquet(grass)

    duration = float(tracks.time_s.max())
    raw_ball = tracks[tracks.cls == "ball"]

    filtered = ball_pitch_filter.filter_ball_by_pitch(tracks, verbose=True)
    scale = pixel_scale.estimate_px_per_m(filtered)

    cam_note = "static, not compensated"
    if profile.compensate_camera:
        motion_path = out_dir / "camera_motion.npy"
        if motion_path.exists():
            motion = np.load(motion_path)
        else:
            print("estimating camera motion...")
            motion = camera_motion.estimate_camera_motion(args.clip)
            np.save(motion_path, motion)
        usable, why = camera_motion.validate_motion(args.clip, motion)
        if usable:
            filtered = camera_motion.compensate(filtered, motion)
            cam_note = f"compensated - {why}"
        else:
            cam_note = f"NOT compensated - {why}"

    before = player_filter.roster_report(filtered)
    capped = player_filter.filter_players(filtered)
    after = player_filter.roster_report(capped)

    selected = ball_selection.select_single_ball(capped, scale, fps=profile.fps)
    diag = ball_selection.ball_track_diagnostics(selected, scale, fps=profile.fps)
    n_before = selected[selected.cls == "player"].track_id.nunique()
    merged = track_reid.merge_fragments(selected, scale, fps=profile.fps)
    n_after = merged[merged.cls == "player"].track_id.nunique()

    metric, absolute = pixel_scale.prepare_tracks_for_events(
        merged, None, verbose=True)
    events = ev_module.detect_events(metric, absolute_pitch=absolute)
    counts = Counter(e["event_type"] for e in events)

    print("\n" + "=" * 74)
    print("OBSERVED BEHAVIOUR  (no ground truth: these are not accuracy)")
    print("=" * 74)

    print(f"\nSCALE & CAMERA")
    print(f"  scale        : {scale:.1f} px/m "
          f"(a 1.75 m player is {scale * 1.75:.0f} px tall)")
    print(f"  camera       : {cam_note}")

    print(f"\nBALL   (expect: exactly one, always)")
    print(f"  candidates   : {len(raw_ball)} over {raw_ball.frame.nunique()} "
          f"frames ({len(raw_ball) / max(raw_ball.frame.nunique(), 1):.2f}/frame)")
    print(f"  selected     : {diag['ball_frames']} frames "
          f"({diag['coverage_pct']:.1f}% coverage)")
    print(f"  speed        : median {diag['median_speed_kmh']:.0f} km/h, "
          f"p95 {diag['p95_speed_kmh']:.0f}  (a struck ball reaches 120)")
    print(f"  size         : ~{profile.ball_size_px:.1f} px across")

    print(f"\nPLAYERS   (expect: 22 on the pitch, plus officials)")
    print(f"  per frame    : {before['median_per_frame']:.0f} -> "
          f"{after['median_per_frame']:.0f} after the roster cap "
          f"(max {after['max_per_frame']})")
    print(f"  track ids    : {n_before} -> {n_after} after re-identification")

    print(f"\nEVENTS   ({duration:.0f} s)")
    print(f"  total        : {len(events)}  {dict(counts)}")
    for kind in ("pass", "carry", "recovery", "tackle"):
        n = counts.get(kind, 0)
        print(f"  {kind:<12} : {n:>3}  ({n / duration * 60:.1f}/min)")
    if not absolute:
        print("  goals, shots and out-of-play are disabled: no homography, so "
              "there is no goal line to compare against")

    ph = stats.physical_stats(metric)
    if not ph.empty and "minutes_tracked" in ph.columns:
        ph = ph[ph.minutes_tracked > 0].copy()
        ph["km_per_90"] = ph.distance_m / ph.minutes_tracked * 90 / 1000
        solid = ph[ph.minutes_tracked >= 0.25]
        print(f"\nPHYSICAL   (expect: 10-12 km per 90 min, top speed 30-36 km/h)")
        print(f"  tracks       : {len(ph)} ({len(solid)} followed >= 15 s)")
        print(f"  per 90 min   : median {ph.km_per_90.median():.1f} km all, "
              f"{solid.km_per_90.median():.1f} km for those >= 15 s")
        print(f"  top speed    : median {ph.top_speed_kmh.median():.1f} km/h, "
              f"max {ph.top_speed_kmh.max():.1f} "
              f"(clipped at MAX_SPEED_KMH = {stats.MAX_SPEED_KMH:.0f})")

        # How much of that is the player. Split each track in half and
        # correlate: a statistic measuring a footballer agrees with itself,
        # one measuring detection jitter does not. Printed rather than
        # assumed, because it depends on the footage -- 0.56 on 720p
        # broadcast, under 0.35 on 640x360.
        rel = stats.top_speed_reliability(metric)
        verdict = ("usable" if rel["r"] >= 0.5 else
                   "WEAK -- do not report per-player speed" if rel["r"] >= 0.35
                   else "NOT MEASURING THE PLAYER -- suppress this figure")
        print(f"  speed trust  : split-half r {rel['r']:.2f} over "
              f"{rel['n']} tracks -- {verdict}")

        spr = stats.sprint_reliability(metric)
        plausible = 0.2 <= spr["per_min"] <= 1.2
        print(f"  sprints      : {ph.n_sprints.sum()} total, "
              f"{spr['per_min']:.2f} per tracked minute "
              f"({'in range' if plausible else 'OUTSIDE'} the 0.3-0.7 "
              "football produces)")
        print(f"  sprint trust : split-half r {spr['r']:.2f} over "
              f"{spr['n']} tracks of 10 s+ -- per-player counts are not "
              "reliable anywhere yet")

    print("\n" + "-" * 74)
    print("These are plausibility checks, not measurements. They catch a")
    print("pipeline that is badly wrong; they cannot tell a good detector from")
    print("a confident one, and this project has been misled by that before.")
    print("Accuracy on this footage needs labels for this footage.")


if __name__ == "__main__":
    main()
