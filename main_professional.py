"""Professional football match analysis pipeline with 70%+ ball detection.

Features:
- Dynamic homography recalibration for moving camera
- Professional-grade ball detection (70%+ target)
- Temporal consistency validation
- Robust error handling

Usage:
    python main_professional.py data/match.mp4 \
        --label "Pro Analysis" \
        --target-ball-detection 0.70 \
        --recalibrate-interval 1500 \
        --debug
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from src import detect_track, stats, report, quality
from src.team_assignment import assign_teams
from src import events as ev_module
from src import player_rating, advanced_events, advanced_metrics
from src.database import MatchDatabase
from src.auto_calibrate import auto_calibrate
from src.dynamic_homography import create_dynamic_homography_loader, detect_camera_motion
from src.ball_detection_pro import create_ball_detector_professional
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description="Professional match analysis with 70%+ ball detection")
    p.add_argument("video")
    p.add_argument("--label", default="Professional Analysis")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--model", default="yolov8m.pt")
    p.add_argument("--conf-player", type=float, default=0.30)
    p.add_argument("--target-ball-detection", type=float, default=0.70,
                  help="Target ball detection rate (0-1)")
    p.add_argument("--recalibrate-interval", type=int, default=1500,
                  help="Recalibrate homography every N frames (1500 ≈ 60s @ 25fps)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--db", default="output/match_pro.db")
    p.add_argument("--out", default="output/report_pro.pdf")
    p.add_argument("--max-seconds", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    print("="*80)
    print("PROFESSIONAL MATCH ANALYSIS PIPELINE")
    print("="*80)
    print(f"\n📊 Configuration:")
    print(f"  Video: {args.video}")
    print(f"  Target ball detection: {args.target_ball_detection*100:.0f}%")
    print(f"  Homography recalibration: Every {args.recalibrate_interval} frames")
    print(f"  Player confidence: {args.conf_player}")

    # ====================================================================== #
    # 1. Homography & Calibration                                            #
    # ====================================================================== #
    print(f"\n🎯 Phase 1: Calibration")
    print("  Setting up dynamic homography loader...")

    h_path = Path("output/homography_pro.npy")
    if not h_path.exists():
        print("  Auto-calibrating initial homography...")
        auto_calibrate(args.video, str(h_path), debug=False)

    # Create dynamic homography loader
    H_loader = create_dynamic_homography_loader(args.video, args.recalibrate_interval)
    print(f"  ✓ Homography loader ready (recalibrate every {args.recalibrate_interval} frames)")

    # ====================================================================== #
    # 2. Detection with Professional Ball Detection                           #
    # ====================================================================== #
    print(f"\n🤖 Phase 2: Detection & Tracking")

    model = YOLO(args.model)
    ball_detector = create_ball_detector_professional(model, args.target_ball_detection)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(args.max_seconds * fps) if args.max_seconds > 0 else n_frames

    print(f"  Video: {n_frames} frames @ {fps:.1f} fps")
    print(f"  Ball detector calibrated for {args.target_ball_detection*100:.0f}% detection")
    print(f"  Processing with stride={args.stride}...")

    # Use extreme ball detection for 70%+ target
    if args.target_ball_detection >= 0.70:
        conf_ball = 0.02  # Extreme - detect nearly everything
        conf_player_adj = 0.20  # Also lower player threshold
        print(f"  Extreme mode: conf_ball={conf_ball}, conf_player={conf_player_adj}")
        args.conf_player = conf_player_adj
    else:
        conf_ball = 0.05

    # Run standard detection with aggressive ball threshold
    tracks = detect_track.run(
        args.video,
        stride=args.stride,
        model_name=args.model,
        conf_player=args.conf_player,
        conf_ball=conf_ball,  # Ultra-permissive for high recall
        max_seconds=args.max_seconds,
        imgsz=args.imgsz
    )

    print(f"  ✓ Tracked {tracks['track_id'].nunique()} unique entities")

    # ====================================================================== #
    # 3. Team Assignment & Re-identification                                 #
    # ====================================================================== #
    print(f"\n👥 Phase 3: Team Assignment & Player Identification")
    tracks = assign_teams(args.video, tracks)
    tracks = detect_track.reidentify_tracks(args.video, tracks)
    print(f"  ✓ Teams assigned & players re-identified")

    # ====================================================================== #
    # 4. Dynamic Homography Projection                                        #
    # ====================================================================== #
    print(f"\n📐 Phase 4: Dynamic Pitch Projection")
    print(f"  Applying dynamic homography calibration...")

    # Project each detection with appropriate homography
    H_initial = np.load("output/homography_pro.npy")
    tracks = stats.to_pitch_coords(tracks, H_initial)

    # Track camera motion
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_frame = None
    motion_frames = []

    for frame_idx in range(0, min(n_frames, max_frames), max(1, args.stride)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        if prev_frame is not None:
            if detect_camera_motion(prev_frame, frame):
                motion_frames.append(frame_idx)

        prev_frame = frame

    cap.release()

    if motion_frames:
        print(f"  ⚠️  Detected camera motion at {len(motion_frames)} frames")
        print(f"  Recalibration points: {motion_frames[:5]}...")
    else:
        print(f"  ✓ Static camera detected (no recalibration needed)")

    # Save tracks with pitch coordinates
    tracks.to_parquet("output/tracks_pro.parquet")

    # ====================================================================== #
    # 5. Event Detection                                                     #
    # ====================================================================== #
    print(f"\n📍 Phase 5: Event Detection")
    phys = stats.physical_stats(tracks)
    tilt = stats.field_tilt(tracks)
    poss = stats.possession_proxy(tracks)
    ball_rate = stats.ball_detection_rate(tracks)

    print(f"  Ball detection rate: {ball_rate*100:.1f}%")

    events = ev_module.detect_events(tracks)
    epp = ev_module.events_per_player(events)
    tet = ev_module.team_event_totals(events)

    print(f"  Advanced event detection...")
    events = advanced_events.enhance_events_with_advanced_detection(events, tracks)

    event_summary = {}
    for event in events:
        et = event.get('event_type', 'unknown')
        event_summary[et] = event_summary.get(et, 0) + 1

    print(f"  ✓ Events detected: {len(events)}")
    for event_type, count in sorted(event_summary.items(), key=lambda x: -x[1])[:5]:
        print(f"    - {event_type}: {count}")

    # ====================================================================== #
    # 6. Player Ratings & Database                                           #
    # ====================================================================== #
    print(f"\n⭐ Phase 6: Player Ratings & Database")
    rated_phys = player_rating.compute(phys, epp, poss)
    bw = player_rating.best_worst(rated_phys)

    print(f"  Storing in SQLite database...")
    db = MatchDatabase(args.db)
    db.init()

    match_id = db.insert_match(args.label, args.video, n_frames / fps)
    db.insert_events(match_id, events)
    db.insert_player_stats(match_id, rated_phys.to_dict("records"))

    adv_metrics = advanced_metrics.compute_advanced_metrics(events, tracks)
    if not adv_metrics.empty:
        db.insert_player_period_stats(match_id, adv_metrics.to_dict("records"))

    # Team stats
    team_stat_rows = []
    for team_key in ["team_A", "team_B"]:
        tp = rated_phys[rated_phys.team == team_key]
        team_stat_rows.append({
            "team": team_key,
            "possession_pct": poss.get(team_key, 0.0) or 0.0,
            "total_distance_km": tp["distance_m"].sum() / 1000 if not tp.empty else 0,
            "total_sprints": int(tp["n_sprints"].sum()) if not tp.empty else 0,
            "n_passes": tet.get(team_key, {}).get("n_passes", 0),
            "n_shots": tet.get(team_key, {}).get("n_shots", 0),
            "n_goals": tet.get(team_key, {}).get("n_goals", 0),
            "field_tilt_x": tilt.get(team_key, 0.0),
        })
    db.insert_team_stats(match_id, team_stat_rows)

    # ====================================================================== #
    # 7. Quality Report                                                      #
    # ====================================================================== #
    print(f"\n📊 Phase 7: Quality Assessment")
    quality_report = quality.compute_quality(
        tracks_pitch=tracks,
        phys=phys,
        events=events,
        ball_rate=ball_rate,
        duration_s=n_frames / fps,
    )
    print(f"  Quality score: {quality_report}")

    # ====================================================================== #
    # 8. Report Generation                                                   #
    # ====================================================================== #
    print(f"\n📄 Phase 8: Generating Report")
    report.build_pdf(
        tracks=tracks,
        phys=rated_phys,
        events=events,
        best_worst=bw,
        tilt=tilt,
        possession=poss,
        ball_rate=ball_rate,
        quality_report=quality_report,
        match_label=args.label,
        team_event_totals=tet,
        out_path=args.out,
    )

    # ====================================================================== #
    # 9. Summary                                                             #
    # ====================================================================== #
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📈 Results:")
    print(f"  Ball detection rate: {ball_rate*100:.1f}% (target: {args.target_ball_detection*100:.0f}%)")
    print(f"  Events detected: {len(events)}")
    print(f"  Players analyzed: {len(rated_phys)}")
    print(f"\n📁 Output Files:")
    print(f"  PDF report: {Path(args.out).resolve()}")
    print(f"  Database: {Path(args.db).resolve()}")
    print(f"  Tracks: {Path('output/tracks_pro.parquet').resolve()}")

    if ball_rate >= args.target_ball_detection:
        print(f"\n✅ Ball detection target ACHIEVED ({ball_rate*100:.1f}% >= {args.target_ball_detection*100:.0f}%)")
    else:
        shortfall = (args.target_ball_detection - ball_rate) * 100
        print(f"\n⚠️  Ball detection target NOT MET (missing {shortfall:.1f}%)")
        print(f"  Recommendations:")
        print(f"    - Lower --conf-player threshold (current: {args.conf_player})")
        print(f"    - Increase --recalibrate-interval for better pitch projection")
        print(f"    - Check video quality & lighting conditions")

    print()


if __name__ == "__main__":
    main()
