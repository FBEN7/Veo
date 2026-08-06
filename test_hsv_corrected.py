#!/usr/bin/env python3
"""HSV detection test - CORRECTED to use pixel coordinates for events.

This version removes corrupted px,py columns before event detection
so that extract_ball_tracking() uses original x,y pixel coordinates.
"""

import shutil
from pathlib import Path
import numpy as np
import pandas as pd

from src.detect_track_hybrid import run as run_hybrid_detection
from src.team_assignment import assign_teams
from src.detect_track import reidentify_tracks
from src import stats, events as ev_module
from src.database import MatchDatabase
from src.auto_calibrate import auto_calibrate

VIDEO_PATH = "/root/.claude/uploads/cd4d7e67-1dd4-5fa1-975c-2f5b3217663b/a382328e-08fd33_4.mp4"
OUTPUT_DIR = Path("output_hsv_corrected")

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("HSV DETECTION TEST - CORRECTED (Using Pixel Coordinates)")
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
    tracks = run_hybrid_detection(
        VIDEO_PATH,
        stride=1,
        model_name="yolov8m.pt",
        conf_player=0.25,
        conf_ball=0.12,
        out_path=str(OUTPUT_DIR / "tracks_hsv.parquet"),
        max_seconds=0,
        imgsz=640,
        ball_detection_method="hsv"
    )
    print(f"  ✓ Tracked {tracks['track_id'].nunique()} unique entities")
    print(f"  ✓ Ball detections: {len(tracks[tracks['cls'] == 'ball'])}")

    # ====================================================================== #
    # 3. Team Assignment & Re-identification                                 #
    # ====================================================================== #
    print(f"\n👥 Phase 3: Team Assignment")
    tracks = assign_teams(VIDEO_PATH, tracks)
    tracks = reidentify_tracks(VIDEO_PATH, tracks)
    print(f"  ✓ Teams assigned & players re-identified")
    print(f"  ✓ Ball detections after assignment: {len(tracks[tracks['cls'] == 'ball'])}")

    # Save with teams
    tracks.to_parquet(str(OUTPUT_DIR / "tracks_with_teams.parquet"))
    print(f"  ✓ Saved final tracks with teams")

    # ====================================================================== #
    # 4. Event Detection (BEFORE pitch projection)                           #
    # ====================================================================== #
    print(f"\n⚽ Phase 4: Event Detection (Pixel Coordinates)")

    # KEY FIX: Remove corrupted px, py columns so extract_ball_tracking()
    # uses the original x, y pixel coordinates instead
    tracks_for_events = tracks.copy()
    if 'px' in tracks_for_events.columns:
        tracks_for_events = tracks_for_events.drop(columns=['px', 'py'])

    print(f"  Using pixel coordinates (removed corrupted px,py)")

    db = MatchDatabase(str(OUTPUT_DIR / "match_hsv.db"))
    db.init()
    match_id = db.insert_match("HSV Test - Pixel Coordinates", VIDEO_PATH)

    # Extract events
    events = ev_module.detect_events(tracks_for_events)
    print(f"  ✓ Events detected: {len(events)}")

    # Breakdown
    event_counts = {}
    for event in events:
        etype = event.get('event_type', 'unknown')
        event_counts[etype] = event_counts.get(etype, 0) + 1

    if len(events) > 0:
        print(f"\n  Event Breakdown:")
        for etype, count in sorted(event_counts.items(), key=lambda x: -x[1]):
            print(f"    {etype:25s}: {count:4d}")

    # Store events
    db.insert_events(match_id, events)
    print(f"\n  ✓ Events stored to database")

    # ====================================================================== #
    # 5. Pitch Projection (players only - after events)                      #
    # ====================================================================== #
    print(f"\n📐 Phase 5: Pitch Projection (Players Only)")
    H_initial = np.load(str(h_path))
    player_tracks = tracks[tracks['cls'] == 'player'].copy()
    player_tracks = stats.to_pitch_coords(player_tracks, H_initial)
    print(f"  ✓ Projected players to pitch coordinates")

    # ====================================================================== #
    # Results                                                                #
    # ====================================================================== #
    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)

    ball_detections = tracks[tracks['cls'] == 'ball']
    ball_frames = len(set(ball_detections['frame']))
    total_frames = len(set(tracks['frame']))
    hsv_detection_rate = ball_frames / total_frames * 100 if total_frames > 0 else 0

    print(f"\nBall Detection Rate:")
    print(f"  HSV: {hsv_detection_rate:.1f}% ({ball_frames}/{total_frames} frames)")
    print(f"  Total detections: {len(ball_detections)}")

    print(f"\nEvent Detection:")
    print(f"  Total events: {len(events)}")
    print(f"  Pass events: {event_counts.get('pass', 0)}")

    if event_counts.get('pass', 0) >= 5:
        print(f"\n  ✅ SUCCESS - {event_counts.get('pass', 0)} passes detected!")
        print(f"     Expected: 5-6 passes")
        print(f"     Result: FIX VERIFIED")
    else:
        print(f"\n  Analysis: {len(events)} events, {event_counts.get('pass', 0)} passes")

    print(f"\n" + "="*80)
    print("✓ Test Complete")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
