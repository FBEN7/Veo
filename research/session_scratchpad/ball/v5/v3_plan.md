# Ball detector v3: training on verified labels (draft plan)

Context: v2 (`.cache/ball_v2.4000.pt`, ResNet-18 to layer2 + lateral + head, stride-4 cells, 6-channel input RGB + motion
D-/D+/F; trained 4,000 steps on 21k weak-label stream strips of M3, M4, M5, M8). Verified labels now exist at the measured frames
of 18 windows: A = M6's 4 windows (human-judged, 546 ball frames of 762), B = 14 windows of M2, M3, M4, M5, M8, M10
(auto-labels from a tile classifier: 8,068 tiles, "B" ~95-100% precise, "not" ~6% balls; plus 600 judged tiles). M7 / M9
sealed for the final test. See v3_understand.md for the code map.

## Data (B windows only; M6 kept out entirely as the selection set)
- Per B frame, the frame's ball: the auto-"B" tiles (judged tiles override auto), merged within 24 px (as `frame_status`);
  frames with B tiles > 24 px apart: dropped. ~1,900 located frames of 3,048.
- Positive strip per located frame: 320 px crop at (t-2, t, t+2, B_t) from the window mp4 (`build_ball_strips.cut` with
  `window_frames`), ball placed with `place()` jitter. Codes: 1 on the cells whose centre is inside the 24 px box around (u, v);
  0 on judged S / B* tiles (spare balls) within a 2-cell disk; -3 on judged N tiles' 3x3 cells; -1 everywhere else (the match
  ball is located, so no other cell is it).
- Negative strip per frame where some tile is a confident non-ball (judged N, or auto "not" with p <= 0.1), the crop centred on
  such a tile: -3 on those tiles' 3x3 cells; in a located frame -1 elsewhere (ball inside the crop -> it is a positive strip
  instead), in an unlocated frame 0 elsewhere.
- Kind "verified" (positive) / "verified_neg"; footprint pooling (as flight: POOL_R 8, FOOTPRINT 4) for the verified bag.

## Training
- Fine-tune from v2.4000 (same architecture, same SIGMA_D / SIGMA_F), 3,000 steps, batch 16: 8 verified, 3 verified_neg, and 5
  of v2's own strips (its MIX_V2 proportions) so the weak-label knowledge of the four v2 matches is kept.
- LR: head/lat 1e-4, conv1_motion 1e-4, layer2 3e-5, early layers 1e-5 (no freeze); warm-up 100, cosine. Same augmentations.
- Monitor holdout: one B window kept out of training (st2_128058_2nd_f015000), monitor only.
- Checkpoints at 1000, 2000, 3000 (`.cache/ball_v3.<step>.pt`), provenance with a new V3_TRAIN set (NEVER keeps 118578, 117092,
  128057, 132831).

## Selection (fixed before training)
- On M6 (A labels; never trained on, images or labels -- though the tile classifier that labelled B saw M6 judgements: disclosed).
- Primary: top-1 hit rate on M6's 546 verified-ball frames (top-1 within 12 px on both axes of the verified ball).
- Rule: best primary among 1000 / 2000 / 3000; the later step when within 0.02.
- Guard: confirmed-false top-1 on M6's unknown frames (top-1 within 6 px of a judged N / S / B* tile) not above v2's + 0.03.
- Diagnostic only: v3 vs v2 on M6 (v2's top-1 from U_a / U_b): biased towards v2 (its top-2 seeded the tiles), and v2's
  checkpoint / Platt / tau were chosen on M6.
- Then: Platt calibration and tau of v3 on M6, as for v2.

## Then (later steps, not now)
- Stride-1 tracking with v3, protocol v5 registered, one-shot test on M7 / M9 with fresh blind tiles (v3 added to the sources).
