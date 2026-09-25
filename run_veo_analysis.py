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

import detect_ball_events as ball_events
import detect_set_pieces as set_pieces
import detect_shots
import score_soccernet as sc
from src import chances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", default="output_veo")
    args = ap.parse_args()

    from src import (auto_tune, pixel_scale, ball_selection, track_reid,
                     ball_pitch_filter, player_filter, camera_motion, stats,
                     ground_plane)
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

    plane = ground_plane.load_or_build(
        args.clip, merged, clip_info["width"], clip_info["height"],
        out_dir / "ground_plane.json")
    calibrated = (ground_plane.effective_px_per_m(merged, plane, verbose=True)
                  if plane is not None else None)

    metric, absolute = pixel_scale.prepare_tracks_for_events(
        merged, None, verbose=True, px_per_m=calibrated)
    events = ev_module.detect_events(metric, absolute_pitch=absolute)
    counts = Counter(e["event_type"] for e in events)

    print("\n" + "=" * 74)
    print("OBSERVED BEHAVIOUR  (no ground truth: these are not accuracy)")
    print("=" * 74)

    print(f"\nSCALE & CAMERA")
    print(f"  scale        : {scale:.1f} px/m "
          f"(a 1.75 m player is {scale * 1.75:.0f} px tall)")
    if calibrated:
        print(f"  calibrated   : {calibrated:.1f} px/m from the ground plane "
              f"-- every distance below is {scale / calibrated:.2f}x what the "
              "height estimate alone gives")
    else:
        print("  calibrated   : NO -- the focal length could not be measured "
              "on this footage, so distances keep the height estimate's bias")
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
        print("  goals and out-of-play are off in HERE: this detector's "
              "coordinates come\n                 from the ground plane, "
              "which has no goal line on it. They are\n                 "
              "detected below instead, on the anchor's maps.")

    # Shots come from the anchor instead, which is a different coordinate
    # source and a measured one.
    #
    # The obvious move is to pass a homography into
    # prepare_tracks_for_events and let `absolute` turn True, and it is the
    # wrong one twice over. That flag also switches on goals and
    # out-of-play, neither of which has ever been validated; and it would
    # hand them the ground plane's coordinates, the component this project
    # measured as making pass F1, speed reliability and distance reliability
    # all worse. It also wants ONE homography for a whole clip, while the
    # anchor is per frame and the camera moves.
    #
    # So shots are detected separately, on the anchor's per-frame maps,
    # which is the path that was actually measured: a planted shot comes
    # back within half a metre on broadcast and its xG within 0.006, and
    # nothing fires on six minutes of verified shot-free football.
    print(f"\nSHOTS")
    try:
        ball = detect_shots.ball_track(out_dir)
        if ball.empty:
            print("  no ball track, so no shots")
        else:
            maps = detect_shots.anchors_for(
                out_dir, clip_info, ball.frame.tolist(),
                np.random.default_rng(0))
            shots, placed = detect_shots.find_shots(ball, maps,
                                                    clip_info["fps"])
            shots = detect_shots.score(
                detect_shots.attribute(shots, out_dir, ball))
            share = placed / max(len(ball), 1)
            print(f"  ball placed  : {placed} of {len(ball)} ball positions "
                  f"({share:.0%}) had a pitch map to sit on")
            print(f"  shots        : {len(shots)}")
            for shot in shots:
                extra = (f", xG {shot['xg']:.3f}" if "xg" in shot else "")
                print(f"    t={shot['time_s']:6.1f}s  "
                      f"{shot['distance_m']:5.1f} m from goal, "
                      f"{shot['speed_ms']:4.0f} m/s{extra}")
            # The bias runs one way and has to be said next to the number.
            if share < 0.35 and shots:
                print("  CAUTION      : at this coverage a shot is usually "
                      "picked up after it\n                 has been "
                      "struck, which reports it closer to goal than it was "
                      "and\n                 so OVERSTATES xG -- measured "
                      "at 0.30 against a true 0.10")
            elif not shots:
                print("  none found. Recall is unmeasured: the labelled "
                      "footage available\n                 contains no "
                      "shots, so this has never been shown to find one it\n"
                      "                 was not given.")

            # Chances created and assists are a join over what is already
            # detected, not a detector: the pass that set up each shot.
            for shot in shots:
                shot["timestamp_s"] = shot["time_s"]
                shot.setdefault("is_goal", False)
            links = chances.link_chances(events, shots)
            tally = chances.summarise(links)
            print(f"  chances      : {tally['chances_created']} created, "
                  f"{tally['assists']} assists")
            if shots and not tally["chances_created"]:
                print("                 no shot had a team-mate's pass "
                      "before it; with no team on\n                 the "
                      "shooter this cannot link, which is a gap rather than "
                      "a zero")

            # Out of play, goals and the restarts that follow, on the same
            # maps -- the anchors are the expensive part and they are
            # already computed.
            stoppages, _ = ball_events.find_ball_events(ball, maps,
                                                        clip_info["fps"])
            goals = [e for e in stoppages if e["event_type"] == "goal"]
            outs = [e for e in stoppages if e["event_type"] == "out_of_play"]
            restarts = set_pieces.find_restarts(ball, maps,
                                                clip_info["fps"], outs)
            kinds = Counter(r["event_type"] for r in restarts)
            print(f"\nBALL OUT OF PLAY")
            print(f"  crossings    : {len(outs)} over the touchline or "
                  f"goal line")
            print(f"  goals        : {len(goals)}")
            print(f"  restarts     : {len(restarts)}  {dict(kinds)}")
            print("  NOTE         : anchors come from the centre circle at "
                  "midfield and the\n                 ball goes out at the "
                  "edges, so coverage where this family of\n                 "
                  "events happens is far below the figure above. On labelled "
                  "footage\n                 one crossing in three is seen; "
                  "the misses are frames with no\n                 pitch map "
                  "at all rather than a geometry error.")
    except Exception as problem:               # never take the run down
        print(f"  unavailable: {problem}")

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
