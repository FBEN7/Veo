#!/usr/bin/env python3
"""Quick test of HSV color-based ball detection vs YOLO.

Compares results on the second 30-second video.
"""
import shutil
import sqlite3
from pathlib import Path

from src.detect_track_hybrid import run as run_hybrid_detection
from src.team_assignment import assign_teams
from src.detect_track import reidentify_tracks
from src import stats
from src.database import MatchDatabase
from src.dynamic_homography import create_dynamic_homography_loader
from src.auto_calibrate import auto_calibrate
from src import events as ev_module
import numpy as np

VIDEO_PATH = "/root/.claude/uploads/cd4d7e67-1dd4-5fa1-975c-2f5b3217663b/863b1c91-08fd33_4.mp4"
OUTPUT_DIR = Path("output_hsv_test")
DB_PATH = OUTPUT_DIR / "match_hsv.db"
REPORT_PATH = OUTPUT_DIR / "report_hsv.pdf"

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("HSV COLOR-BASED BALL DETECTION TEST")
    print("="*80)
    print(f"\nVideo: {VIDEO_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    # ====================================================================== #
    # 1. Calibration                                                         #
    # ====================================================================== #
    print(f"\n🎯 Phase 1: Calibration")
    h_path = OUTPUT_DIR / "homography_hsv.npy"
    if not h_path.exists():
        print("  Auto-calibrating initial homography...")
        auto_calibrate(VIDEO_PATH, str(h_path), debug=False)
    print(f"  ✓ Homography ready")

    # ====================================================================== #
    # 2. Detection with HSV Ball Detection                                   #
    # ====================================================================== #
    print(f"\n🤖 Phase 2: Detection & Tracking (HSV Method)")
    print(f"  Ball detection: HSV color-based (v2 with circularity filtering)")
    print(f"  Processing...")

    tracks = run_hybrid_detection(
        VIDEO_PATH,
        stride=1,
        model_name="yolov8m.pt",
        conf_player=0.25,
        conf_ball=0.12,
        out_path=str(OUTPUT_DIR / "tracks_hsv.parquet"),
        max_seconds=0,
        imgsz=640,
        ball_detection_method="hsv"  # <-- HSV detection!
    )

    print(f"  ✓ Tracked {tracks['track_id'].nunique()} unique entities")

    # ====================================================================== #
    # 3. Team Assignment & Re-identification                                 #
    # ====================================================================== #
    print(f"\n👥 Phase 3: Team Assignment")
    tracks = assign_teams(VIDEO_PATH, tracks)
    tracks = reidentify_tracks(VIDEO_PATH, tracks)
    print(f"  ✓ Teams assigned & players re-identified")

    # ====================================================================== #
    # 4. Pitch Projection                                                     #
    # ====================================================================== #
    print(f"\n📐 Phase 4: Pitch Projection")
    H_initial = np.load(str(h_path))
    tracks = stats.to_pitch_coords(tracks, H_initial)
    print(f"  ✓ Projected to pitch coordinates")

    # ====================================================================== #
    # 5. Event Detection                                                      #
    # ====================================================================== #
    print(f"\n⚽ Phase 5: Event Detection")

    db = MatchDatabase(str(DB_PATH))
    db.init()
    match_id = db.insert_match("HSV Test - Second Video", VIDEO_PATH)

    # Extract events
    events = ev_module.extract_all_events(tracks)
    print(f"  Total events detected: {len(events)}")

    # Breakdown by type
    event_counts = {}
    for event in events:
        etype = event.get('event_type', 'unknown')
        event_counts[etype] = event_counts.get(etype, 0) + 1

    print(f"\n  Event Breakdown:")
    print(f"  " + "-"*40)
    for etype, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        print(f"    {etype:25s}: {count:4d}")
    print(f"  " + "-"*40)

    # Store events
    db.insert_events(match_id, events)

    print(f"\n  ✓ Events stored to {DB_PATH}")

    # ====================================================================== #
    # Summary & Comparison                                                    #
    # ====================================================================== #
    print(f"\n" + "="*80)
    print("COMPARISON: YOLO vs HSV Detection")
    print("="*80)

    # Get ball detection stats
    ball_detections = tracks[tracks['cls'] == 'ball']
    ball_frames = len(set(ball_detections['frame']))
    total_frames = len(set(tracks['frame']))
    hsv_detection_rate = ball_frames / total_frames * 100 if total_frames > 0 else 0

    print(f"\nBall Detection:")
    print(f"  YOLO (previous run): 42.8% (321/750 frames)")
    print(f"  HSV (this run):      {hsv_detection_rate:.1f}% ({ball_frames}/{total_frames} frames)")
    print(f"  Improvement:         +{hsv_detection_rate - 42.8:.1f}%")

    print(f"\nEvent Detection:")
    print(f"  YOLO (previous run): 139 events")
    print(f"    - 138 false positives (shot_inferred)")
    print(f"    - 1 real event (out_of_play)")
    print(f"  HSV (this run):      {len(events)} events")
    for etype, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        print(f"    - {count} {etype}")

    print(f"\n" + "="*80)
    print("✓ Test Complete")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
