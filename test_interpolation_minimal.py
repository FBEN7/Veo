#!/usr/bin/env python3
"""Minimal test to verify ball interpolation works in the pipeline.

Tests on just the first 300 frames (~12 seconds @ 25fps) to complete quickly.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

from src import detect_track
from src.ball_interpolation import interpolate_ball_track, get_ball_detection_continuity

def main():
    video_path = "data/match_first2min.mp4"

    if not Path(video_path).exists():
        print(f"Error: {video_path} not found")
        return

    print("="*60)
    print("MINIMAL BALL INTERPOLATION TEST")
    print("="*60)

    # Verify video is readable
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"\n📹 Video: {n_frames} frames @ {fps:.1f} fps ({n_frames/fps:.1f}s)")

    # Run detection on ALL frames (since video is only 2 min)
    print(f"\n🤖 Phase 1: Detection (all {n_frames} frames)...")
    tracks = detect_track.run(
        video_path,
        stride=1,  # Process every frame
        model_name="yolov8m.pt",
        conf_player=0.15,  # Ultra-low player threshold
        conf_ball=0.01,    # Maximum extreme ball detection
        max_seconds=0,     # No limit
        imgsz=640
    )

    print(f"\n✓ Detection complete: {len(tracks)} total detections")
    print(f"  Unique entities: {tracks['track_id'].nunique()}")

    # Analyze ball detection continuity BEFORE interpolation
    print(f"\n📊 Before Ball Interpolation:")
    continuity_before = get_ball_detection_continuity(tracks)
    print(f"  Total frames in video: {continuity_before['total_frames']}")
    print(f"  Frames with ball detection: {continuity_before['frames_with_detection']}")
    print(f"  Detection rate: {continuity_before['detection_rate']*100:.1f}%")
    print(f"  Average gap size: {continuity_before['avg_gap_size']:.1f} frames")
    print(f"  Max gap size: {continuity_before['max_gap_size']} frames")
    print(f"  Number of gaps: {continuity_before['num_gaps']}")

    # Apply ball interpolation
    print(f"\n🔄 Phase 2: Ball Interpolation (max_gap_frames=5)...")
    tracks_interpolated = interpolate_ball_track(tracks, max_gap_frames=5)

    # Analyze continuity AFTER interpolation
    print(f"\n📊 After Ball Interpolation:")
    continuity_after = get_ball_detection_continuity(tracks_interpolated)
    print(f"  Frames with ball detection: {continuity_after['frames_with_detection']}")
    print(f"  Detection rate: {continuity_after['detection_rate']*100:.1f}%")
    print(f"  Average gap size: {continuity_after['avg_gap_size']:.1f} frames")
    print(f"  Max gap size: {continuity_after['max_gap_size']} frames")
    print(f"  Number of gaps: {continuity_after['num_gaps']}")

    # Calculate improvement
    improvement_frames = continuity_after['frames_with_detection'] - continuity_before['frames_with_detection']
    improvement_pct = (continuity_after['detection_rate'] - continuity_before['detection_rate']) * 100

    print(f"\n✨ Improvement:")
    print(f"  Additional frames filled: +{improvement_frames}")
    print(f"  Detection rate improvement: +{improvement_pct:.1f}%")

    # Save results
    output_path = "output/test_interpolation_minimal.parquet"
    Path("output").mkdir(exist_ok=True)
    tracks_interpolated.to_parquet(output_path)
    print(f"\n✓ Interpolated tracks saved to {output_path}")

    print("\n" + "="*60)
    print("✅ TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
