#!/usr/bin/env python3
"""Test ball interpolation on existing track data."""

import pandas as pd
from pathlib import Path
from src.ball_interpolation import interpolate_ball_track, get_ball_detection_continuity

def main():
    tracks_path = "output/tracks.parquet"

    if not Path(tracks_path).exists():
        print(f"Error: {tracks_path} not found")
        return

    print("="*70)
    print("BALL INTERPOLATION TEST ON EXISTING TRACKS")
    print("="*70)

    # Load existing tracks
    print(f"\n📁 Loading tracks from {tracks_path}...")
    tracks = pd.read_parquet(tracks_path)
    print(f"  ✓ Loaded {len(tracks)} detections")

    # Analyze structure
    print(f"\n📊 Track Statistics:")
    print(f"  Columns: {list(tracks.columns)}")
    print(f"  Total rows: {len(tracks)}")
    print(f"  Frame range: {tracks['frame'].min()} to {tracks['frame'].max()}")

    # Check if we have ball detections
    if 'cls' in tracks.columns:
        ball_mask = tracks['cls'] == 'ball'
        print(f"  Ball detections: {ball_mask.sum()}")

    # Get ball detection continuity BEFORE interpolation
    print(f"\n🔍 Before Ball Interpolation:")
    continuity_before = get_ball_detection_continuity(tracks)
    print(f"  Total frames: {continuity_before['total_frames']}")
    print(f"  Frames with ball: {continuity_before['frames_with_detection']}")
    print(f"  Detection rate: {continuity_before['detection_rate']*100:.1f}%")
    print(f"  Number of gaps: {continuity_before['num_gaps']}")
    print(f"  Average gap size: {continuity_before['avg_gap_size']:.1f} frames")
    print(f"  Max gap size: {continuity_before['max_gap_size']} frames")

    # Apply interpolation
    print(f"\n🔄 Applying ball interpolation (max_gap=5 frames)...")
    tracks_interp = interpolate_ball_track(tracks, max_gap_frames=5)

    # Get continuity AFTER interpolation
    print(f"\n🔍 After Ball Interpolation:")
    continuity_after = get_ball_detection_continuity(tracks_interp)
    print(f"  Total frames: {continuity_after['total_frames']}")
    print(f"  Frames with ball: {continuity_after['frames_with_detection']}")
    print(f"  Detection rate: {continuity_after['detection_rate']*100:.1f}%")
    print(f"  Number of gaps: {continuity_after['num_gaps']}")
    print(f"  Average gap size: {continuity_after['avg_gap_size']:.1f} frames")
    print(f"  Max gap size: {continuity_after['max_gap_size']} frames")

    # Calculate improvement
    print(f"\n✨ Improvement Summary:")
    frames_added = continuity_after['frames_with_detection'] - continuity_before['frames_with_detection']
    detection_improvement = (continuity_after['detection_rate'] - continuity_before['detection_rate']) * 100
    gap_reduction = continuity_before['num_gaps'] - continuity_after['num_gaps']

    print(f"  Frames filled: +{frames_added}")
    print(f"  Detection rate improvement: +{detection_improvement:.1f}%")
    print(f"  Gaps reduced: {gap_reduction} (from {continuity_before['num_gaps']} to {continuity_after['num_gaps']})")

    # Save interpolated tracks
    output_path = "output/tracks_interpolated.parquet"
    tracks_interp.to_parquet(output_path)
    print(f"\n✓ Interpolated tracks saved to {output_path}")

    # Check for interpolated flag
    if 'interpolated' in tracks_interp.columns:
        interpolated_count = tracks_interp['interpolated'].sum()
        print(f"  Interpolated rows added: {interpolated_count}")

    print("\n" + "="*70)
    print("✅ INTERPOLATION TEST SUCCESSFUL")
    print("="*70)

if __name__ == "__main__":
    main()
