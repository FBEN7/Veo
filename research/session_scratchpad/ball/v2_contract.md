# Ball detector v2 + events v3: implementation contract

Repo /home/user/Veo, branch claude/game-analysis-veo-8kze2z. The design is in
scratchpad/ball/design_v2.json (results[-2] = plan with label_spec, model_spec,
training_spec, inference_spec, protocol_v3, events_protocol; results[-1] = critique).
THIS FILE OVERRIDES the plan where they differ.

## Global rules for implementers
- Edit ONLY the files you own (listed per component). Read anything.
- Keep every v1 path reproducible: existing CLIs and defaults behave exactly as now
  (v1 checkpoint .cache/ball_heatmap.3000.pt must load with build() and give identical
  outputs; eval_ball_detector.py without new flags gives identical numbers).
- Match the repo's style: module docstring explaining the method, terse comments,
  `from src import soccertrack_v2 as st`, `from src.paths import DATA_DIR`.
- Every new module gets a `_check()` (synthetic, no network, < 60 s) runnable as
  `python -c "from X import _check; _check()"` or `python X.py --check`.
- No heavy compute: never stream a match half, never train > 30 steps, never run a
  detector over more than ~20 frames. CPU is 4 cores shared with other implementers.
- Never print/commit tokens; never commit; never write into data/ except via
  fetch_soccertrack_v2.py (do not run it). Do not open matches 128057 (M7), 132831 (M9).
- Frame indices: BAS events (st.events) and GSR (st.gsr_table) use ABSOLUTE 0-based
  frames within a half. Window files (data/soccertrack_v2/<w>.mp4, _gt.parquet,
  _events.csv) use LOCAL frames; <w>.json has match, half, start_frame, width, height,
  pitch (outline polygon in image px). absolute = start_frame + local.

## Fixed decisions (protocol v3, with the critique's fixes)
- Train v2: both halves of 118575, 118576, 118577, 128058. In the 2nd halves of
  118576 and 128058 skip absolute frames 13500-20999. Never cut: 118578 (U),
  117093/132877 (V), 117092 (M1), 128057/132831 (test).
- U (all choices): windows st2_118578_1st_f030000, st2_118578_2nd_f015000 (exist),
  st2_118578_1st_f052500, st2_118578_2nd_f037500 (to cut). T diagnostic only:
  st2_118576_2nd_f015000, st2_128058_2nd_f015000 (need --allow-seen).
  V-old: st2_117093_1st_f015000, st2_132877_1st_f015000. V-new (cut after freeze):
  st2_117093_1st_f045000, st2_117093_2nd_f015000, st2_117093_2nd_f045000,
  st2_132877_1st_f045000, st2_132877_2nd_f015000, st2_132877_2nd_f045000.
- Motion: LUMINANCE ONLY, 3 channels (critique fix). grey = cv2 BGR2GRAY float.
  D- = (g_t - g_{t-2})/255/SIGMA_D, D+ = (g_t - g_{t+2})/255/SIGMA_D,
  F = (g_t - B_t)/255/SIGMA_F. Missing t-2/t+2 -> use t (D = 0). Missing B -> F = 0.
- Background B_t (absolute frame t): centre c = 50*round(t/50); samples = absolute
  frames that are multiples of 50 within [c-350, c+350] and inside the available
  range [first, last] (half for training, window for inference); B = per-pixel
  median of the grey samples; needs >= 8 samples else None. Record n_samples.
- Strip = one JPEG q98, 1280x320 = [t-2 | t | t+2 | B_t] crops at the same (x0, y0),
  BGR; B stored grey replicated to 3 channels.
- Measures: FF-auto (4.0 m from segment A->B, projection t in [0,1], |t-s| <= 0.30,
  >= 1.0 m from every GSR person; ground point = image_to_pitch(u, v + size/2);
  pass score = mean of its 4 frames). Placebo pre-kick (kick-15, kick-30 for passes
  whose previous event is >= 35 frames before the kick and is a DRIVE by the passer
  or a PASS received by the passer): hit = within 4 m of the upcoming segment,
  t in [0.2, 0.9], >= 1 m from persons. Placebo post-reception (critique fix):
  frame = reception + 15 (if inside the window and before the next event): hit =
  within 4 m of the just-completed segment, t in [0.1, 0.7], >= 1 m from persons.
- Statistics: paired per-pass differences; 1-minute block bootstrap (blocks =
  window x minute of the pass's kick), 4000 draws, seed 0; one-sided alpha 0.025:
  "lower bound" = 2.5th percentile; reject when it is > 0.

## Component B — model and motion (owner of: src/motion_bg.py NEW, src/ball_heatmap.py)
src/motion_bg.py:
- constants BG_STEP = 50, BG_HALF_SPAN = 350, BG_MIN = 8, OFFSET = 2.
- `bg_centre(t) -> int`; `bg_sample_frames(t, first, last) -> list[int]`.
- `class BackgroundCache(first, last, crop=None)`: `.add(abs_frame, bgr_or_grey)` stores
  grey (optionally cropped to crop=(x0,y0,x1,y1)) only for frames that are multiples of
  BG_STEP; `.get(t) -> (grey_uint8 | None, n_samples)` caches the median per centre;
  `.ready(t) -> bool` true once every sample frame <= min(c+350, last) was added;
  `.drop_before(abs_frame)` frees memory.
- `motion_channels(prev_bgr, cur_bgr, nxt_bgr, bg_grey) -> float32 HxWx3` (D-, D+, F
  normalised with ball_heatmap.SIGMA_D / SIGMA_F; None inputs per the rules above).
- `class RingBuffer(n=5)` for sequential decode: push(abs_frame, bgr); get(abs_frame).
- `_check()`.
src/ball_heatmap.py (keep v1 API and build() default identical):
- `build(motion: bool = False)`: motion adds `conv1_motion = nn.Conv2d(3, 64, 7, 2, 3,
  bias=False)` zero-initialised; its output is added to conv1(rgb) before bn1. Input
  tensor: N x 6 x H x W (3 RGB ImageNet-normalised + 3 motion) when motion, else N x 3.
  v1 state_dicts must still load into build().
- `SIGMA_D`, `SIGMA_F` module constants (placeholders 0.02 / 0.05; component C
  measures and writes them later via a documented one-line edit — NOT C's file; C
  prints the values and the main agent edits).
- `to_tensor_v2(rgb_list, motion_list) -> tensor N x 6 x H x W`.
- `detect_v2(model, prev, cur, nxt, bg, outline, tau=0.05, top_k=8, margin_px=10, cal=None,
  motion_zero=False) -> np.ndarray rows (u, v, p_raw, size, logit)`; same crop/peaks/
  outline/refinement as v1 detect(); size = BALL_M * max(sx, sy) via local_scale when cal
  given else nan; prev/nxt/bg may be None; motion_zero=True sets the motion input to 0
  (diagnostic). bg is the full-frame grey B or a crop matching the pitch-bbox crop: accept
  either (document which).
- `load_calibration(ckpt) -> dict | None` from `<ckpt>.calib.json` ({"a","b","tau",...});
  `p_cal(logit, calib) -> sigmoid(a*logit+b)`.
- `write_provenance(ckpt, info: dict)`, `read_provenance(ckpt) -> dict | None`
  (`<ckpt>.provenance.json`), `V2_TRAIN = {"118575": (1, 2), "118576": (1, 2),
  "118577": (1, 2), "128058": (1, 2)}`, `V2_EXCLUDE = {("118576", 2): (13500, 20999),
  ("128058", 2): (13500, 20999)}`, `NEVER = {"118578", "117093", "132877", "117092",
  "128057", "132831"}`, and `check_unseen(ckpt, match, allow_seen=False)` raising if the
  match is in the checkpoint's training provenance.
- `_check()` covering: v1 build/load unchanged, motion model with zero motion equals RGB
  path output, detect_v2 vs detect on a synthetic frame with motion_zero.

## Component C — strips and labels (owner of: build_ball_strips.py NEW, label_ball_strips.py NEW)
Do not modify build_ball_patches.py / build_ball_labels.py (v1); import helpers from them.
build_ball_strips.py:
- `python build_ball_strips.py --match M --half H --out DIR [--kinds ...] [--seed 0]`
  streams the half (st.open_video, sequential decode) and, alternatively,
  `--window W --out DIR` reads a local window mp4 (absolute = start_frame + local;
  background range = window). Refuse matches not in V2_TRAIN unless --window with
  --allow-any (smoke tests / the mismatch check only); apply V2_EXCLUDE.
- Sample kinds per label_spec (plan): flight (completed ground PASS and intercepted PASS,
  s in {0.25,0.5,0.75}+U(-0.08,0.08) clipped [0.17,0.83], t_hat = s + 0.4 s(1-s)),
  lofted (HIGH PASS, CROSS at s in {0.35,0.65}), action (v1 KEEP_ACTION + FREE KICK +
  THROW IN), dribble (midpoint of gaps > 1.5 s opened by a DRIVE), free (2 per every
  second action frame; categories cluster 35% / landmark 25% / boundary 25% / random 15%;
  every crop cell >= 12 m from actor and track point; up to 10 resamples).
  Crop placement: jitter per label_spec (slack-aware, <= 100 px). The crop must be inside
  the frame.
- Queue each sample at decode time with its t-2/t/t+2 crops; finish it once the
  BackgroundCache is ready for t; write `<kind>_<match>_<half>_<frame>_<x0>_<y0>.jpg` and
  append metadata rows to `strips.parquet` (match, half, frame, kind, category, x0, y0, A,
  B (pitch metres, as 4 columns), s, t_hat, L, event ids, n_bg). Memory: keep only grey
  background samples (pitch-bbox crop) + queued crops.
- Needs per-frame people (GSR) only for labels, not here (except free-category picks,
  which may use the half's GSR via st.gsr_table — acceptable; cached parquet).
label_ball_strips.py:
- `python label_ball_strips.py --strips DIR [--coco] [--holdout 300]` -> labels.parquet
  (file, match, half, frame, kind, category, x0, y0, positive_only, weight, holdout) and
  labels_codes.npy (N x 80 x 80 int8): codes per label_spec (corridor bag with height,
  ignore band 6 m / 48 px + 2 m curtain, hard negatives = every body >= 1 m from the bag
  incl. passer/receiver + landmark disks 0.75 m (>= 12 px) outside the ignore band;
  >= 85% of bag inside the crop on a 640 px canvas else retry window else drop; lofted
  curtain; action = v1 rule; dribble positive-only weight 0.5; free = -1, bodies -3,
  landmarks -3, COCO sports-ball (yolov8s conf >= 0.25 on the t crop upsampled x2) not
  within 1 m of a landmark -> 1 m (>= 12 px) ignore disk).
  Landmarks (pitch metres, centre origin): centre spot, penalty spots (+-41.5, 0),
  corners, goal posts (+-52.5, +-3.66), box corners (+-36, +-20.16) and (+-52.5, +-20.16),
  6-yard corners (+-47, +-9.16) and (+-52.5, +-9.16), halfway/touchline crossings (0, +-34).
- `--sigma` measures SIGMA_D / SIGMA_F (robust std, e.g. 1.4826*MAD of the D and F
  channels over 1,000 strips' pitch pixels) and prints them; also prints label stats.
- `--compare-window W --strips-a DIR_A --strips-b DIR_B` (mismatch check): per-channel D/F
  statistics of strips built from the source stream vs from the local window at the same
  frames; prints the D-scale ratio.
- Label constants recorded in a dict LABEL_CONSTANTS (printed into provenance).
- Recompute and print (do not change constants) the T-only statistics of the confirmed
  in-flight balls from scratchpad tiles_ff (T) judgements: perpendicular distance to the
  segment (m), pixel distance, t - s, nearest-player distance.

## Component D — training (owner of: train_ball_heatmap.py)
- Add `--strips DIR` (v2 path) next to the v1 `--patches` path (unchanged). `--motion` /
  `--no-motion`, `--steps 4000`, checkpoints 2500/3250/4000 (`<out>.<step>.pt`) with
  provenance JSON (training matches/halves, excluded ranges, LABEL_CONSTANTS, SIGMA
  values, recipe hash = sha256 of this file + label_ball_strips.py, seed).
- Loader: split the strip JPEG into 4 crops; augmentation per training_spec (identical
  flip/colour/scale across the 4 crops; JPEG re-encode p 0.3 one q in [80,95] per crop;
  per-crop gaussian noise sigma 0-2; temporal reversal p 0.5; motion dropout p 0.1 ->
  prev = nxt = bg = cur); compute motion channels with motion_bg.motion_channels.
- Batch 16: flight 5, action 3, dribble 1, lofted 1, free 6 (landmark 2, cluster 2,
  boundary 1, random 1); fall back to any free if a category is short.
- Loss per training_spec: flight bag pool z = (1/8) log((1/4) sum exp(8 z)) with BCE to 1;
  action/dribble v1 log-mean-exp r=5 (dribble weight 0.5, positive-only); focal negatives
  gamma 2, x3 on -3; peak suppression: BCE(z,0) on the 16 largest 5x5 local maxima among
  -1/-3 cells, weight 1/16 each; sum over patches / 16. Drop-worst from step 1200: flight
  25%, action 15%, dribble 25%.
- Optimiser per training_spec (conv1_motion from step 1 at 3e-4; rgb conv1/bn1/layer1
  frozen to 800 then 3e-5; layer2 1e-4; head+lat 3e-4; warm-up 100; cosine to 0).
- Monitor every 250 steps on holdout strips: bag hit, peakiness (mean bag cells with
  p > 0.5; abort if > 30 after step 1500), negative peaks p > 0.15 per patch,
  conv1_motion weight norm. Log to stdout.
- `--bench N`: time N steps (no saving) for the go/no-go; prints s/step.

## Component E — calibration and dense detection (owner of: calibrate_ball_heatmap.py NEW, detect_ball_heatmap.py)
- calibrate_ball_heatmap.py `--scores U.json --ckpt CKPT --out CKPT.calib.json`: Platt
  (a, b) by logistic regression on the saved top-8 logits at U pass frames (positive if
  the candidate meets the FF-auto geometry for that pass frame) and PASS/DRIVE frames
  (positive if within 1.5 m of the actor); writes a, b, reliability table (10 bins), ECE,
  tau = smallest p_cal on 0.05..0.95 step 0.05 with <= 1.0 candidates per frame on U
  random frames. Reads candidate/geometry data from the scores JSON written by
  eval_ball_detector.py (component A) — agree on the JSON fields below.
- detect_ball_heatmap.py: add `--v2` (motion model; pass 1 grabs background samples, pass 2
  sequential decode with RingBuffer), `--calib` default `<ckpt>.calib.json`, rows with
  confidence = p_cal, extra column logit, detection_method "heatmap_v2", keep p_raw >= 0.05,
  <= 8 per frame (`--top-k`), `--stride 2`, `--extra-frames` (local frames to add),
  `--allow-seen`; refuse training-provenance matches otherwise. v1 path unchanged.

## Component A — measurement (owner of: eval_ball_detector.py, validate_ff_auto.py NEW, ball_tiles.py)
eval_ball_detector.py (v1 behaviour and registered outputs unchanged):
- new measures FF-auto, placebo-pre, placebo-post (definitions above) for every detector,
  per window and pooled; per-pass arrays saved; per-match breakdown.
- `compare_ff(results, a, b)`: paired 1-minute block bootstrap (4000 draws, seed 0),
  returns mean diff, 2.5th / 97.5th percentiles, n passes, per-match diffs.
- `--v2 CKPT [CKPT...]` motion detectors (one decode per window; background pass first);
  `--motion-zero` adds the zeroed diagnostic for each v2 ckpt; `--heatmap` (v1) and COCO
  as now. Save top-8 rows WITH logits for v2 ("candidates": {frame: [[u,v,p_raw,size,
  logit], ...]}); v1/COCO rows as now.
- frames: frames_of() unchanged + "placebo_pre": [{frame, pass index}], "placebo_post":
  [...]; the scores JSON results[*] gets "passes" detail list: for each pass {from, to,
  frames, s values, minute block, kick frame, reception frame}, and "actions" PASS/DRIVE
  frames with actor x,y, "random" frames — enough for calibrate_ball_heatmap.py to
  recompute geometry without the video.
- `--reuse SCORES.json [...]`: recompute all measures from saved candidates without
  decoding (detectors present in the file); must reproduce V.json's registered 0.23/0.23.
- `--candidates NAME=PARQUET` : take a detector's rows from a dense tracks parquet
  (detect_ball_heatmap.py output) instead of running it.
- `--sealed OUT`: write everything to OUT and print only frame counts and timings (no
  measure of any kind) — also suppress the legacy summary.
- `--allow-seen`: required for windows whose match is in a v2 checkpoint's provenance.
validate_ff_auto.py (new): from scratchpad tiles_T, tiles_ff, tiles_ffV, tiles_E4
(judged.txt + key.json) and score files (heatmap_T.json, coco_T2.json, V.json), compute
FF-auto's agreement with the judge (kappa, PPV, recall) on pass-frame tiles per set, decoys
counted as hits, chance rate (random pitch points), and the FF-auto table for v1 and COCO
on T and V-old (per window, pooled, paired diff with the block bootstrap). Also
`--sealed-check S.json --judged DIR` for the S2 validity conditions (kappa per detector,
differential bias, decoys).
ball_tiles.py (existing behaviour unchanged): add `--pass-frame q` (q-th of the 4 frames,
1-based, one tile per pass per detector), `--corridor-decoys N` (on the pass segment at
t = s +- 0.40 within [0.05, 0.95], perpendicular offset U(-1.5, 1.5) m, >= 1 m from
persons, >= 30 px from every detector's candidates), `--dedupe-px 6` (merge identical
picks of several detectors into one tile credited to each), `--rejudge KEYFILE:N` (re-show
N old tiles), `--confident NAME=TAU:N` (N confident picks on random frames), key records
pass ids.

## Component F — events (owner of: src/ball_link.py NEW, src/events.py, score_soccernet.py, sweep_events_v2.py NEW, score_events_paired.py NEW)
- src/events.py: `detect_events(..., params: dict | None = None)` temporarily overrides
  module constants from a WHITELIST (POSSESSION_RADIUS_M, MIN_POSSESSION_HOLD_S,
  CARRY_MIN_TIME_S, MIN_PASS_DISTANCE_M, SHOT_MIN_KMH, SHOT_MAX_DIST_FROM_GOAL_M,
  SHOT_MOUTH_MARGIN_M); unknown names raise; restored in finally. New
  SHOT_MOUTH_MARGIN_M = None (v1 behaviour): when set, a shot needs the ball's ray to
  cross the goal line within +-(3.66 + margin) m (in _toward_goal or where shots are
  decided). Defaults leave every output identical.
- src/ball_link.py: `link(rows, cal, players, c_null=1.0, c_static=2.0, fps=25)` skip-
  Viterbi in pitch metres per inference_spec (unary -logit(p_cal)+logit(tau); measurement
  sigma 1.5 px / min(sx, sy) clipped [0.05, 1.0] m; speed free to 35 m/s, cost
  4 (v/35-1)^2 above, forbidden > 60; skips <= 5 detection frames cost c_null each; static
  penalty for a candidate within 0.5 m for >= 2 s and >= 3 m from every pipeline player)
  -> one ball row per frame (same columns as input rows). `_check()` with synthetic tracks.
- score_soccernet.py: `swap_ball_rows(src_dir, dst_dir, rows)` (logic of scratchpad e5.py:
  copy clip.json, profile.json, pitch.json, team_features.json; replace ball rows in
  tracks.parquet and tracks_teams.parquet; players first within a frame);
  `events_from_cached(run_dir, cal, select="v1"|"link", link_params=None, top_k=None,
  tau=None, event_params=None) -> (events, merged)`: factor the post-team part of
  run_pipeline (grass filter, filter_ball_by_pitch, inside_pitch, select_single_ball OR
  link, merge_fragments (players keep merge_fragments in both paths; assert the player
  and team rows are identical across selections via a hash), to_pitch_metres,
  detect_events(absolute_pitch=True, ball_time_aware=True, params=event_params)).
  run_pipeline output unchanged for existing callers. Regression: events_from_cached on
  output_<V-old>/events_coco reproduces its events.json exactly (do not print scores).
- score_events_paired.py: pooled pass+carry F1 at 0.5/1/2 s per configuration dir, chance
  F1 from 200 random-time draws per window, paired 10 s block bootstrap 4000 draws seed 0,
  one-sided alpha 0.025 (2.5th percentile); `--configs a b` for paired difference;
  reproduces scratchpad e5_score.py numbers (0.353 / 0.442) on events_coco / events_heatmap.
- sweep_events_v2.py: staged grid per events_protocol E2 on U-E windows
  (st2_118578_1st_f030000, st2_118578_2nd_f015000) from cached run dirs with a given ball
  rows parquet per window; objective E3; G-E pooled with leave-one-window-out reported;
  writes events_v3.json (+ sha256) — do not run it on V.
