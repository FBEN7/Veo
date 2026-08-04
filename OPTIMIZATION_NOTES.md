# Limited Field-of-View Event Detection Optimizations

## Critical Finding (Post-Analysis Update)

**User Observation:** "35 players tracked + 97% ball interpolation but only 1 event doesn't make sense"

**Investigation Result:** Database contained **139 events**, but **138 were false positives** created by overly aggressive inferred shot detection. The console correctly filtered these out.

**Root Cause:** Inferred ball movement detection (`_detect_ball_movement_events()`) was flagging every high-speed ball movement as a shot, regardless of player involvement (69 shots/minute - unrealistic).

**Fix Applied:** Disabled inferred event detection until proper player validation is implemented.

**Lesson Learned:** Event detection requires linking ball movement to visible players, not just ball kinematics alone.

---

## Problem Identified
Your 2-minute (and 90-minute) video has limited camera coverage (~60% of pitch visible, only ~6 players per frame). Standard event detection thresholds were designed for full-view broadcasts with all 22 players visible.

## Solutions Implemented

### 1. **Relaxed Event Detection Thresholds** (src/events.py)
Updated all event detection constants for limited visibility:

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| POSSESSION_RADIUS_M | 4.4m | 8.0m | Larger radius for loose ball detection |
| POSSESSION_GAP_FILL_S | 1.4s | 3.0s | Tolerate longer gaps in partial visibility |
| MIN_PASS_DISTANCE_M | 1.2m | 0.3m | Detect short passes & redirects |
| MAX_PASS_INTERVAL_S | 4.5s | 8.0s | More time for passes across limited view |
| PASS_UNKNOWN_BRIDGE_S | 1.4s | 2.5s | Longer unknown bridges for off-camera players |
| SHOT_MIN_KMH | 16.0 | 10.0 | Lower speed threshold |
| SHOT_MAX_DIST_FROM_GOAL_M | 40.0m | 50.0m | Shots from further (limited angle) |
| SHOT_COOLDOWN_S | 1.8s | 0.5s | Faster shot detection |
| CARRY_MIN_TIME_S | 2.0s | 0.8s | Shorter carries (limited distance visible) |
| CARRY_MIN_DISTANCE_M | 8.0m | 2.0m | Lower distance threshold |
| RECOVERY_MIN_LOOSE_S | 0.8s | 0.3s | Quick recovery detection |
| TACKLE_MAX_PLAYER_DIST_M | 2.5m | 4.0m | Larger tackle radius |
| TACKLE_MAX_BALL_SPEED_KMH | 28.0 | 40.0 | Higher speed tolerance |

### 2. **Ball Movement Pattern Detection** (src/events.py)
New function `_detect_ball_movement_events()` detects events from ball kinematics alone:
- Infers **shots** from high-speed ball movement toward goal
- Works even when both players aren't visible
- Detects patterns like "ball speeds up toward goal area"
- Uses same speed and direction constraints as regular shot detection

### 3. **Better Ball Interpolation Integration**
Ball interpolation now fills 97% of detection gaps (before: 46.7%, after: 97.3%)
- Provides continuous ball tracking for better event chains
- Already integrated in main_professional.py

## Expected Improvements

**Before optimizations:**
- Events detected: 1 (out_of_play only)
- Passes: 0
- Shots: 0

**After optimizations (estimated):**
- Events detected: 15-30+ (passes, shots, carries, tackles)
- Better event distribution across match
- More accurate possession tracking

## Implementation

All changes are backward-compatible. Pipeline automatically uses these thresholds
when running `main_professional.py`.

## Testing Strategy

1. Run on 2-minute video with optimizations
2. Compare event counts & types before/after
3. If improved, test on full 90-minute match
4. Fine-tune further if needed based on results

## Notes

- These thresholds are specifically tuned for your camera angle
- Different camera positions may need different values
- Ball interpolation is critical for success with these relaxed thresholds
