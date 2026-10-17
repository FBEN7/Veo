# Protocol v4 (draft): tracking the ball across frames

Question: does linking v2's per-frame candidates over time (and filling short gaps)
put the ball in the right place more often than v2's raw top candidate, in free flight,
on matches never used for any choice?

## Fixed inputs (frozen, not tuned here)
- Detector: ball_v2.4000.pt, its U calibration (a 0.895, b -0.928, tau 0.25).
- Dense candidates: detect_ball_heatmap.py --v2 --stride 2 (top-8, p_raw >= 0.05, confidence = p_cal).
- Linker: src/ball_link.link (pitch-metre skip-Viterbi, null state, static penalty), top-k 4.
- Measured frames: eval_ball_detector.frames_of (4 frames per completed ground pass at s = 0.25,
  0.417, 0.583, 0.75; placebo pre/post; PASS/DRIVE; random).

## The tracker (src/ball_track.py, new)
1. Rows with p_cal >= tau, top-k per frame -> ball_link.link -> at most one pick per detection frame.
2. Gap fill: between consecutive picks a (frame fa) and b (fb) with fb - fa <= G frames, every frame
   strictly between gets (u, v, size) linearly interpolated in the picture, source "interpolated";
   picks keep source "detected". No extrapolation beyond the first/last pick of a chain.
3. Players: none (the static penalty ignores player clearance): no player pipeline is needed, so the
   tracker runs on any window.
4. The tracked pick at a measured frame = the track's position there, else (variant F) the raw top-1
   at that frame from the dense rows if the frame is a detection frame, else none (a miss).
Odd measured frames are therefore always interpolated (stride 2): that is the deployed setting.

## Splits
- Tuning: U-E windows st2_118578_1st_f030000, st2_118578_2nd_f015000 (dense parquets exist; 24 passes).
- U-test (sanity gate): st2_118578_1st_f052500, st2_118578_2nd_f037500 (39 passes; dense to run).
- Report: V-new, the 6 windows registered in protocol v3 and never cut:
  st2_117093_1st_f045000, st2_117093_2nd_f015000, st2_117093_2nd_f045000,
  st2_132877_1st_f045000, st2_132877_2nd_f015000, st2_132877_2nd_f045000 (~108 passes).
  V-old (seen by v2) is not used. M7/M9 never opened.

## Tuning on U-E (only)
Grid: c_null {0.5, 1, 2} x c_static {0, 2, 6} x G {6, 12} frames x F {no, yes} = 36 configurations.
Objective: FF-auto (tracked) pooled over the 24 U-E passes; constraint: placebo-pre and placebo-post
each <= max(0.05, raw v2's on the same frames + 0.03). Ties within 0.02: the default (c_null 1,
c_static 2, G 6, F no) wins, then fewer changes from it.

## Gate on U-test (before V-new is cut)
All of: (a) FF-auto tracked - raw >= 0 (point); (b) placebos within the constraint above;
(c) blind sitting S-U: one tile per U-test pass at frame index 1 for tracked and raw (deduped within
6 px, credited to both) + 10 random + 10 corridor decoys: tracked match-ball rate >= raw's, decoys
judged ball <= 2 of 20. If the gate fails: report U results, no V-new.

## Claims on V-new (one shot; sealed until the sitting is judged)
- H5 (primary, blind): at frame index 1 of every V-new pass, the tracked v2 pick is the match ball
  in play (inside the 24 px box) more often than v2's raw top-1: exact one-sided McNemar, p < 0.025,
  more tracked-only hits. (Blind is primary because FF-auto rewards any point on the pass line, which
  is where interpolation puts the ball.)
- H6 (blind, only if H5 holds): tracked v2 vs COCO's raw top-1, same test.
- Secondary: FF-auto paired (tracked - raw, tracked - COCO), 1-minute block bootstrap; placebos;
  share of tracked picks that are interpolated; per match; decoys.
- Sitting S-V: tracked v2, raw v2, raw COCO at index 1 per pass (deduped within 6 px), 12 random +
  12 corridor decoys, seed 4; key read only after judging.

## Disclosures
- One seed, one detector. FF-auto validity for interpolated picks unknown (hence blind primary).
- V-new is the last unseen M2/M10 data; using it here spends it.
