#!/usr/bin/env python3
"""Diagnostic script to analyze detection quality in early frames.

Usage:
    python debug_early_frames.py data/match.mp4 [--max-frames 300]
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from src import detect_track
from tqdm import tqdm


def debug_early_detection(video_path: str, max_frames: int = 300, stride: int = 1):
    """Analyze detection quality in early frames."""
    print(f"\n📊 Early Frame Diagnostic Report")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Analyzing: first {max_frames} frames\n")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"Frame rate: {fps} FPS")
    print(f"First {max_frames} frames = {max_frames/fps:.1f} seconds\n")

    # Run detection on early frames
    from ultralytics import YOLO
    import supervision as sv

    model = YOLO("yolov8m.pt")
    tracker = sv.ByteTrack(
        frame_rate=fps/stride,
        track_activation_threshold=0.08,
        lost_track_buffer=180,
        minimum_matching_threshold=0.78,
    )

    frame_stats = []
    frame_idx = 0
    pbar = tqdm(total=min(max_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
                desc="Analyzing")

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % stride == 0:
            # Run detection
            res = model(frame, verbose=False, conf=0.12, imgsz=640,
                       classes=[0, 32])[0]
            det = sv.Detections.from_ultralytics(res)

            players = det[det.class_id == 0]
            if len(players) > 0:
                players = players[players.confidence >= 0.25]
            players = tracker.update_with_detections(players)

            balls = det[det.class_id == 32]
            if len(balls) > 0:
                balls = balls[balls.confidence >= 0.12]

            frame_stats.append({
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "n_players": len(players),
                "n_player_detections": len(det[det.class_id == 0]),
                "n_unique_track_ids": len(np.unique(players.tracker_id)) if len(players) > 0 else 0,
                "n_balls": len(balls),
                "avg_player_confidence": float(players.confidence.mean()) if len(players) > 0 else 0.0,
                "avg_ball_confidence": float(balls.confidence.mean()) if len(balls) > 0 else 0.0,
            })

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    stats_df = pd.DataFrame(frame_stats)

    # Print analysis
    print("\n📈 Detection Statistics:")
    print(f"{'-'*60}")
    print(f"Total frames analyzed: {len(stats_df)}")
    print(f"Time span: {stats_df['time_s'].max():.1f} seconds")
    print(f"\nPlayer Detection:")
    print(f"  Mean players/frame: {stats_df['n_players'].mean():.1f}")
    print(f"  Min players/frame:  {stats_df['n_players'].min()}")
    print(f"  Max players/frame:  {stats_df['n_players'].max()}")
    print(f"  Mean confidence:    {stats_df['avg_player_confidence'].mean():.3f}")

    print(f"\nTrack Stability:")
    print(f"  Mean unique IDs/frame: {stats_df['n_unique_track_ids'].mean():.1f}")
    print(f"  (High variance = track ID jumping)")

    print(f"\nBall Detection:")
    print(f"  Frames with ball: {(stats_df['n_balls'] > 0).sum()} / {len(stats_df)}")
    print(f"  Detection rate: {(stats_df['n_balls'] > 0).sum() / len(stats_df) * 100:.1f}%")
    if (stats_df['n_balls'] > 0).any():
        print(f"  Mean confidence: {stats_df[stats_df['n_balls'] > 0]['avg_ball_confidence'].mean():.3f}")

    # Identify problem frames
    print(f"\n⚠️  Problem Frames:")
    low_players = stats_df[stats_df['n_players'] < 8]
    if len(low_players) > 0:
        print(f"  Frames with <8 players: {len(low_players)} ({len(low_players)/len(stats_df)*100:.1f}%)")
        if len(low_players) <= 10:
            print(f"    Frames: {low_players['frame'].tolist()}")

    high_id_variance = stats_df['n_unique_track_ids'].std()
    if high_id_variance > 2.0:
        print(f"  High track ID variance: σ={high_id_variance:.1f} (indicates frequent re-identification)")
        problematic = stats_df[stats_df['n_unique_track_ids'] < stats_df['n_unique_track_ids'].mean() - 2]
        print(f"    Frames with unusually few IDs: {len(problematic)}")

    low_ball = stats_df[stats_df['n_balls'] == 0]
    if len(low_ball) / len(stats_df) > 0.5:
        print(f"  Ball detection <50%: expect event detection to fail")

    # Print frame-by-frame for first 30 frames
    print(f"\n📋 Frame-by-frame (first 30 frames):")
    print(f"{'-'*60}")
    print(stats_df[['frame', 'time_s', 'n_players', 'n_unique_track_ids', 'n_balls']].head(30).to_string(index=False))

    return stats_df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    debug_early_detection(args.video, max_frames=args.max_frames, stride=args.stride)
