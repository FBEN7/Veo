#!/usr/bin/env python3
"""Phase 2: Advanced Event Detection and Metrics Validation."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("="*80)
print("PHASE 2: ADVANCED EVENT DETECTION & METRICS TEST")
print("="*80)

# Create realistic synthetic data with more complex scenarios
np.random.seed(42)

num_frames = 500  # 20 seconds at 25 FPS
fps = 25
duration = num_frames / fps
frame_nums = np.arange(num_frames)
time_s = frame_nums / fps

print(f"\nPhase 2 Test Setup:")
print(f"  Duration: {duration:.1f}s ({num_frames} frames)")
print(f"  Test Scenarios: Passes, dribbles, pressures, shots, clearances")

# ============================================================================
# Create Complex Ball Movement
# ============================================================================
print("\n" + "="*80)
print("SCENARIO 1: POSSESSION AND PASSING")
print("="*80)

ball_tracks = []
player_tracks = []

# Simulate possession sequence with passes
possession_sequences = [
    {"team": "team_A", "start_frame": 0, "duration": 150, "start_x": 20, "start_y": 34},
    {"team": "team_B", "start_frame": 150, "duration": 100, "start_x": 80, "start_y": 34},
    {"team": "team_A", "start_frame": 250, "duration": 200, "start_x": 50, "start_y": 34},
]

# Create ball and player tracks
for seq in possession_sequences:
    team = seq["team"]
    start_frame = seq["start_frame"]
    duration = seq["duration"]
    start_x = seq["start_x"]

    for i in range(duration):
        frame = start_frame + i
        if frame >= num_frames:
            break

        t = frame / fps
        # Ball trajectory within possession
        progress = i / duration
        bx = start_x + progress * 30  # Move along field
        by = 34 + 5 * np.sin(i * 0.05)

        # Add small noise
        bx += np.random.normal(0, 0.2)
        by += np.random.normal(0, 0.2)

        if np.random.random() > 0.15:  # 85% detection
            ball_tracks.append({
                'frame': int(frame),
                'time_s': float(t),
                'cls': 'ball',
                'px': float(np.clip(bx, 0, 105)),
                'py': float(np.clip(by, 0, 68)),
            })

    # Players for this team - close to ball
    if team == "team_A":
        player_positions = [
            (bx - 5, 30), (bx - 5, 38), (bx + 5, 34), (bx + 15, 34)
        ]
        team_id = "team_A"
    else:
        player_positions = [
            (bx + 5, 30), (bx + 5, 38), (bx - 5, 34), (bx - 15, 34)
        ]
        team_id = "team_B"

    for frame in range(start_frame, min(start_frame + duration, num_frames)):
        t = frame / fps
        for pid, (px_base, py_base) in enumerate(player_positions):
            px = px_base + np.random.normal(0, 1)
            py = py_base + np.random.normal(0, 1)

            if np.random.random() > 0.20:
                player_tracks.append({
                    'frame': int(frame),
                    'time_s': float(t),
                    'cls': 'player',
                    'track_id': pid + (1 if team_id == "team_A" else 12),
                    'team': team_id,
                    'px': float(np.clip(px, 0, 105)),
                    'py': float(np.clip(py, 0, 68)),
                })

# ============================================================================
# Add Other Players (Defending)
# ============================================================================
print("Adding defensive players...")

for frame in frame_nums:
    t = frame / fps

    # Team B defends when Team A has ball
    if 0 <= frame < 150 or 250 <= frame < 450:
        for pid in range(8):  # 8 defending players
            px = 70 + np.random.normal(0, 5)
            py = 20 + pid * 5 + np.random.normal(0, 2)

            if np.random.random() > 0.20:
                player_tracks.append({
                    'frame': int(frame),
                    'time_s': float(t),
                    'cls': 'player',
                    'track_id': 20 + pid,
                    'team': 'team_B',
                    'px': float(np.clip(px, 0, 105)),
                    'py': float(np.clip(py, 0, 68)),
                })

    # Team A defends when Team B has ball
    if 150 <= frame < 250:
        for pid in range(8):  # 8 defending players
            px = 35 + np.random.normal(0, 5)
            py = 20 + pid * 5 + np.random.normal(0, 2)

            if np.random.random() > 0.20:
                player_tracks.append({
                    'frame': int(frame),
                    'time_s': float(t),
                    'cls': 'player',
                    'track_id': 30 + pid,
                    'team': 'team_A',
                    'px': float(np.clip(px, 0, 105)),
                    'py': float(np.clip(py, 0, 68)),
                })

all_tracks = pd.concat(
    [pd.DataFrame(ball_tracks), pd.DataFrame(player_tracks)],
    ignore_index=True
).sort_values('frame').reset_index(drop=True)

print(f"Ball detections: {len(ball_tracks)}")
print(f"Player detections: {len(player_tracks)}")

# ============================================================================
# Run Event Detection Pipeline
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: EVENT DETECTION PIPELINE")
print("="*80)

from src.events import detect_events, _ball_kinematics, _possession_per_frame

# Step 1: Ball Kinematics
print("\n[STEP 1] Ball Kinematics")
ball_kin = _ball_kinematics(all_tracks)
print(f"  Ball frames: {len(ball_kin)}")

# Step 2: Possession
print("\n[STEP 2] Possession Tracking")
poss = _possession_per_frame(ball_kin, all_tracks)
with_possessor = poss[poss['possessor_id'] != -1]
possession_rate = len(with_possessor) / len(poss) if len(poss) > 0 else 0
print(f"  Possession detection rate: {possession_rate*100:.1f}%")

# Step 3: Event Detection
print("\n[STEP 3] Basic Event Detection")
events = detect_events(all_tracks)
print(f"  Total events: {len(events)}")

event_counts = {}
for e in events:
    et = e.get('event_type', 'unknown')
    event_counts[et] = event_counts.get(et, 0) + 1

for et in sorted(event_counts.keys()):
    print(f"    {et:20s}: {event_counts[et]:3d}")

# ============================================================================
# Step 4: Advanced Event Detection
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: ADVANCED EVENT DETECTION")
print("="*80)

from src import advanced_events

print("\n[STEP 4] Advanced Event Detection")
enhanced_events = advanced_events.enhance_events_with_advanced_detection(events, all_tracks)
print(f"  Events after enhancement: {len(enhanced_events)}")

adv_event_counts = {}
for e in enhanced_events:
    et = e.get('event_type', 'unknown')
    adv_event_counts[et] = adv_event_counts.get(et, 0) + 1

print(f"\n  Event breakdown:")
for et in sorted(adv_event_counts.keys()):
    print(f"    {et:20s}: {adv_event_counts[et]:3d}")

# Count new events added
new_events = len(enhanced_events) - len(events)
print(f"\n  Events added by advanced detection: {new_events}")

# ============================================================================
# Step 5: Advanced Metrics
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: ADVANCED METRICS")
print("="*80)

from src import advanced_metrics

print("\n[STEP 5] Computing Advanced Metrics")
adv_metrics = pd.DataFrame()  # Initialize empty dataframe
try:
    adv_metrics = advanced_metrics.compute_advanced_metrics(enhanced_events, all_tracks)
    if not adv_metrics.empty:
        print(f"  Player metrics computed: {len(adv_metrics)} players")
        print(f"  Columns: {list(adv_metrics.columns)}")

        # Show sample metrics
        print(f"\n  Sample metrics:")
        numeric_cols = adv_metrics.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            vals = adv_metrics[col].dropna()
            if len(vals) > 0:
                print(f"    {col:30s}: mean={vals.mean():.2f}, max={vals.max():.2f}")
    else:
        print("  ⚠️  No metrics generated")
except Exception as e:
    print(f"  ❌ Error computing metrics: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Validation
# ============================================================================
print("\n" + "="*80)
print("PHASE 2 VALIDATION")
print("="*80)

checks = {
    "Ball Kinematics": len(ball_kin) > 0,
    "Possession Detection": possession_rate > 0.4,
    "Basic Events": len(events) > 0,
    "Advanced Detection": len(enhanced_events) > len(events),
    "Advanced Metrics": len(adv_metrics) > 0 if not adv_metrics.empty else False,
}

print("\nValidation Results:")
all_pass = True
for check, result in checks.items():
    status = "✓" if result else "✗"
    print(f"  [{status}] {check}")
    all_pass = all_pass and result

# Event type validation
expected_events = {"pass", "carry", "recovery"}
found_events = set(event_counts.keys())
has_expected = expected_events.issubset(found_events)
print(f"  [{'✓' if has_expected else '✗'}] Expected event types detected")

# Advanced events validation
advanced_expected = {"dribble", "pressure", "clearance"}
found_advanced = set(adv_event_counts.keys())
has_advanced = len(advanced_expected & found_advanced) > 0
print(f"  [{'✓' if has_advanced else '✗'}] Advanced events detected")

print("\n" + "="*80)
if all_pass and has_expected:
    print("✅ PHASE 2 VALIDATION PASSED")
    print("\nKey achievements:")
    print(f"  ✓ Ball kinematics working ({len(ball_kin)} frames)")
    print(f"  ✓ Possession tracking at {possession_rate*100:.1f}%")
    print(f"  ✓ Basic events detected ({len(events)} events)")
    print(f"  ✓ Advanced events enhanced (+{new_events} events)")
    print(f"  ✓ Advanced metrics computed ({len(adv_metrics)} players)")
    print("\n✓ Ready to proceed to Phase 3 (Database & Reporting)")
else:
    print("❌ PHASE 2 VALIDATION FAILED")
    print("\nIssues:")
    for check, result in checks.items():
        if not result:
            print(f"  ✗ {check}")
    if not has_expected:
        print(f"  ✗ Missing expected event types")
    if not has_advanced:
        print(f"  ✗ No advanced events detected")

print("="*80)

# Summary statistics
print("\nPhase 2 Summary Statistics:")
print(f"  Test Duration: {duration:.1f}s")
print(f"  Ball Detections: {len(ball_kin)}")
print(f"  Possession Rate: {possession_rate*100:.1f}%")
print(f"  Basic Events: {len(events)}")
print(f"  Enhanced Events: {len(enhanced_events)}")
print(f"  Event Types: {len(adv_event_counts)}")
print(f"  Player Metrics: {len(adv_metrics)}")
