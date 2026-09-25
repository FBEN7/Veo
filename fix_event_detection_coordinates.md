# Critical Bug Found: Event Detection Coordinate Mismatch

## Problem
- **Possession changes detected:** 88 (good!)
- **Pass events recorded:** 0 (bad!)
- **Root cause:** Ball kinematics are in PIXEL coordinates, but pass validation expects PITCH METERS

## Why Passes Are Filtered Out
```python
if travel < MIN_PASS_DISTANCE_M or dt_change > MAX_PASS_INTERVAL_S:
    return  # Skips pass event
```

With:
- `MIN_PASS_DISTANCE_M = 0.3` meters
- Ball coordinates in PIXELS (e.g., 100-1900 range)
- Travel distance between frames in pixels (e.g., 2-50 pixels)

**2-50 pixels ≈ 0.1-0.5 meters** - but this varies by video scale!

Most passes have travel < 0.3 meters and get filtered out.

## Solution
Project ball coordinates to pitch space (meters) BEFORE running event detection.

Current flow:
```
Phase 4: Pitch Projection (players only)
Phase 5: Event Detection (ball in pixels + players in meters) ❌
```

Fixed flow:
```
Phase 4: Pitch Projection (players AND ball)
Phase 5: Event Detection (all in pitch meters) ✓
```

## Implementation
Need to pass the homography transform to event detection and convert ball coordinates to pitch space using the same transform used for players.
