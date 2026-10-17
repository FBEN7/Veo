## Revised protocol (v4-r1)

### Question
Does linking v2's candidates over time, with short gap fill, make v2's pick the match ball in play more often than v2's raw top candidate at the same frame, in free flight, on matches never used for any choice?

### Fixed inputs (nothing tuned)
- **Detector:** `ball_v2.4000.pt` (sha256 31f9a5eb…), with its U calibration: a 0.895072, b −0.927959, tau 0.25 (sha256 77799641…).
- **Tracker input:** `detect_ball_heatmap.py --v2 --stride 2` with no `--extra-frames` (top-8, p_raw ≥ 0.05, confidence = p_cal). `ball_track` refuses a sidecar with extra frames, or with any frame not divisible by the stride.
- **Comparators at measured frames:**
  - Raw v2 and COCO come from the `eval_ball_detector.py --v2` decode: `U_a.json` / `U_b.json` for U, a sealed run for V-new.
  - Its v2 rows must equal the dense rows at shared frames (a count).
- **Measured frames:** `frames_of`, as registered.

### Tracker (`src/ball_track.py`)
Code and config sha256 recorded before U-test.
1. Take rows with p_cal ≥ 0.25, top 4 per frame.
2. Call `ball_link.link(rows, cal, empty players, c_null=1, c_static=2, tau=0.25, frames=the sidecar's processed frames)`. No players and no ground truth; the static penalty applies to any candidate that stayed within 0.5 m for 2 s or more.
3. **Fill:** between consecutive picks at fa < fb with fb − fa ≤ 6, interpolate (u, v, size) linearly in the picture; source "interpolated". No extrapolation.
4. **Fallback:** a processed frame (even) with no pick and no fill gets the dense raw top-1 at any p; source "fallback".
5. Otherwise the frame has no pick (a miss). An odd frame is never a detection frame.
6. **Output:** one row per covered frame, with a source column. Confidence is p_cal for a detection or fallback, and the lower endpoint's p_cal for an interpolated pick. The sidecar lists every local frame as processed.

**Why these values:**
- The linker constants are the ones frozen for events, and on U-E they change at most 3 of 135 measured picks.
- G = 6 cannot bridge the middle half of any measured pass (passes are at least 20 frames).
- The fallback keeps the tracker's coverage at least equal to raw's at detection frames.

### Splits
- **Dry run, U-E** (`st2_118578_1st_f030000`, `st2_118578_2nd_f015000`; 24 passes): counts and implementation checks only. No choice depends on it.
- **Harm/futility gate, U-test** (`st2_118578_1st_f052500`, `st2_118578_2nd_f037500`; 39 passes). This split is not held out for the detector.
- **Report, V-new:** `st2_117093_1st_f045000`, `st2_117093_2nd_f015000`, `st2_117093_2nd_f045000`, `st2_132877_1st_f045000`, `st2_132877_2nd_f015000`, `st2_132877_2nd_f045000` (about 108 passes).
- V-old is not used. M7 and M9 are never opened.

### Pairs and sittings
- **Pair frames:** every pass frame (q0–q3) where tracked and raw v2 are more than 6 px apart or exactly one is empty, plus q1 of every pass.
- **Tiles:**
  - Tracked and raw v2 at the pair frames, deduped within 6 px. COCO at q1 on V-new only.
  - Decoys:
    - random;
    - corridor;
    - offset: a tracked pick moved 18–20 px in a random direction, same frame, at least 13 px from every v2 candidate;
    - re-judged: old tiles from `tiles_ffV`.
  - The key records source, q, parity, tracked source and local px/m.
- **Judging:**
  - The ball counts if the centre of the match ball in play is inside the 24 px box. Unsure counts as not.
  - Record the sha256 of `judged.txt` before reading `key.json`.

### Gate on U-test (all must hold; otherwise report U-test and do not cut V-new)
- **G-a:** passes with d > 0 outnumber passes with d < 0. This is the H5 statistic, as a point estimate.
- **G-b:** tracked placebo hits ≤ raw's + 1, separately for pre-kick and post-reception.
- **G-c:** random plus corridor decoys judged the ball ≤ 2 of 20, and offset decoys ≤ 1 of 6.
- **G-d:** no fill longer than 6 frames, no "detected" or "fallback" pick at an odd frame, and the input sidecar is stride-only.

Recorded but not gated:
- b and c (frames and passes) and the implied V-new power;
- FF-auto agreement (kappa) on interpolated tracked tiles;
- confirmed false FF-auto hits.

### Claims on V-new (one shot, sealed until judged)
- **H5 (primary, blind), tracked v2 against raw v2:**
  - Per pass, d = (pair frames where only tracked is the ball) − (frames where only raw is).
  - Exact one-sided sign test over passes with d ≠ 0, p < 0.025, with more passes where d > 0.
- **H6 (only if H5 holds):** tracked v2 against COCO's raw top-1 at q1, exact one-sided McNemar, p < 0.025.
- **Secondary:**
  - q1-only McNemar, tracked against raw.
  - The decomposition by tracked source, parity, match, and near/far.
  - FF-auto paired, tracked − raw, with a 1-minute block bootstrap. It counts as a measure only if kappa on interpolated tracked tiles is at least 0.6 and tracked's confirmed false hits are at most raw's + 2; otherwise it is reported as invalid.
  - Placebos; share of tracked picks that are interpolated or fallback; decoys; re-judge agreement.
- **Sitting S-V:** seed 4; 12 random, 12 corridor and 12 offset decoys, and 12 re-judged tiles.
- **Expected size (U-E counts):** about 63 informative passes, against about 27 at q1 alone. Power is about 0.75 at a 70/30 split, against about 0.4 for q1 alone.

### Order
1. Write and self-check `ball_track` on synthetic data, then freeze the hashes.
2. U-E dry run (counts only).
3. U-test:
   - finish the dense runs already going (stride 2, no extras);
   - run the tracker;
   - sealed `eval --reuse U_b.json --candidates track=…`;
   - S-U tiles, judging, key, gate.
4. V-new:
   - cut the windows;
   - dense run at stride 2;
   - sealed `eval --v2` with COCO and the tracker;
   - check the rows are identical at shared frames (a count);
   - S-V tiles, judging, key, tests.

### Disclosures
- One seed and one detector. The Platt fit used FF-auto labels at U pass frames, so U numbers are optimistic; V-new is unaffected.
- The linker constants were chosen on U-E for events, with players; here it runs without them.
- GT-free counts on U-E were seen before registration.
- V-old events with v2 and the linker were seen, and V-new lies in the same halves.
- A win means tracking at stride 2 beats detection at the same frame. At odd frames raw has a detection the deployed tracker never had, which is conservative for the tracker.
- V-new is the last unseen M2/M10 data, and this spends it.

## Amendments after the implementation review (adopted before anything is frozen)
- Gap fill on the pitch: the ground points of the two picks are interpolated linearly in pitch metres and projected back
  (pitch_to_image), v = ground v - size/2 with size interpolated linearly; a fill is made only when the implied ground speed
  between the two picks is <= 35 m/s (the linker's free speed) and fb - fa <= 6 (always < (MAX_SKIP + 2) x stride, so never across
  a linker restart).
- src/ball_link.py stays as committed (its sha256 is in events_v3.json); ball_track passes an empty players table.
- The tracker refuses a dense sidecar with extra frames or frames off the stride; the sealed log counts dense top-1 vs the eval
  decode's top-1 at shared frames; a measured frame past the last processed frame is measured as none (counted).
- Frozen tracker parameters: tau 0.25, top_k 4, c_null 1, c_static 2, gap 6, fill speed <= 35 m/s, fallback on; params file + sha256,
  code sha256 (ball_track.py, ball_link.py, eval_ball_detector.py) recorded before U-test.

## Amendments at freeze (implementation, before any U-test tracker output exists)
- Invocation: the tracker runs inside eval, `eval_ball_detector.py --reuse U_b.json --track T=.cache/v2_balls_{window}.parquet
  --track-params track_params.json --sealed ...` (not `--candidates track=...`); the tracked rows are the run's results, with the
  per-frame source in `results[w]["T"]["track source"]`; no tracked parquet. The tracked detector is named `T` throughout.
- `track_params.json` = {"frozen": true, "params": {tau 0.25, top_k 4, c_null 1.0, c_static 2.0, gap 6, max_fill_speed 35.0,
  fallback true}} (sha256 0ecf6095...), written by `tune_ball_track.py`.
- Fill and fallback output: a frame past the last processed one by less than the stride is measured as none (counted); any other
  uncovered measured frame stops the run.
- S-U: `ball_tiles.py --pairs T ball_v2.4000.pt --decoys 10 --corridor-decoys 10 --offset-decoys 6 --seed 4`; offset decoys keep
  >= 13 px from every candidate of every detector in the scores file (stricter than registered); a pass frame that either detector did
  not measure stops the sitting; a frame the video lacks stops it (no blank tiles). G-c checks the registered counts (20, 6), so it
  applies to S-U only.
- G-d also requires every window's sidecar to carry the registered checkpoint (31f9a5eb...) and calibration (77799641...) sha256.
- The "interpolated or fallback" share is over pass frames with a tracked pick (none excluded).
- The re-judged decoys come from `tiles_ffV` (V-old tiles): re-judging old labels reads no new measure.
