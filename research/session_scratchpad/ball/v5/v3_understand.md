

# read:model

Report: the v2 ball model and how it runs at inference (code read, plus one quick forward-pass timing)

**1. Architecture** (`src/ball_heatmap.py`, `build()` L70-114)
- The backbone is torchvision ResNet-18 with `IMAGENET1K_V1` weights, cut after layer2. Layout: `stem` = conv1 7x7/2, bn1, relu, maxpool, then `layer1` (64 ch, stride 4), then `layer2` (128 ch, stride 8).
- `lat` is a 1x1 conv (128->64) on layer2. Its output is upsampled bilinearly to layer1's size and added to layer1.
- `head` is Conv3x3 64->64 (no bias), BN, ReLU, then Conv1x1 64->1 with bias initialised to -4.6.
- Output is `[N, H/4, W/4]` logits: `STRIDE = 4` (L43), so one cell is 4x4 px. There are 737,793 parameters.
- Motion stem: `conv1_motion = Conv2d(3, 64, 7, 2, 3, bias=False)`, initialised to zero. When the input has 6 channels, it is fused with the RGB conv1 into one 6-channel conv (L101-105) before bn1. A 3-channel input goes through the RGB stem only, so it behaves exactly as the RGB net.
- Input normalisation: BGR is converted to RGB, divided by 255, then ImageNet `MEAN`/`STD` (L44-45). Channels 3-5 are the motion channels.
- Training input is 320x320 crops (`train_ball_heatmap.py` L229 `SIZE, STRIDE = 320, 4`), so the code grid is 80x80. A strip is a 1280x320 JPEG at q98 laid out as `[t-2 | t | t+2 | B_t]`, read by `read_strip` (L339).

**2. Inference input on a full panorama** (`detect_v2`, L296-354)
- **No tiling and no 320 px strips at inference.** Frames are 4096x1080, for both windows and stream.
  - `pitch_box` (L174) crops to the pitch outline's bounding box: 32 px wider on each side, 64 px higher (`pad_above`) and 32 px lower (`pad_below`).
  - On the windows I checked, that gives a crop of about 657x3890. It is zero-padded at the bottom and right to a multiple of 8 (e.g. 664x3896).
  - One forward pass over the whole crop gives a logit map of about 166x974.
- The input is built in place by `_input_v2` (L265) as 1x6xHxW: normalised RGB plus `motion_bg.motion_planes` (L187):
  - `D- = (g_t - g_{t-2}) / 255 / sigma_d`
  - `D+ = (g_t - g_{t+2}) / 255 / sigma_d`
  - `F = (g_t - B_t) / 255 / sigma_f`
  - g is cv2 BGR2GRAY. `OFFSET = 2` frames. A missing t±2 or B_t gives a zero channel.
- Background B_t (`motion_bg`, L46-63):
  - It is the per-pixel median (`median_u8`) of grey frames at multiples of `BG_STEP = 50`, within `BG_HALF_SPAN = 350` of the centre `c = 50 * round(t / 50)`, kept inside `[first, last]`.
  - Fewer than `BG_MIN = 8` samples gives None. 15 is a full set (`FULL_BG`, eval L227).
  - `BackgroundCache(first, last, crop=pitch_box)` stores grey crops.
  - At inference `[first, last]` is the window (start_frame .. start_frame + frames - 1), so the first and last ~350 frames have fewer samples. Training strips used the whole half.
- `motion_zero=True` feeds the 3-channel input. This is how RGB-only checkpoints run.

**3. Checkpoint loading** (`load_v2`, L449-484)
- It reads the state dict with `torch.load(map_location="cpu")` and `read_provenance(ckpt)` from `<ckpt>.provenance.json`.
- `rgb_only` means `"conv1_motion.weight" not in state`. If the provenance's `"motion"` flag contradicts the weights, it raises.
- It always builds `build(motion=True)`:
  - A motion checkpoint loads strictly.
  - An RGB-only one loads with `strict=False`, and only `conv1_motion.weight` may be missing.
- It returns `(model.eval(), {"rgb_only", "sigma_d", "sigma_f"})`. The sigmas come from provenance `"sigma"` via `provenance_sigma` (L438). A motion checkpoint without them raises.
- The module constants are `SIGMA_D = 0.0535` and `SIGMA_F = 0.0174`, with `SIGMA_MEASURED = True` (L56-59). Training reads them at call time; inference uses the provenance values.
- `ball_v2.4000.pt` provenance:
  - `sigma` {0.0535, 0.0174}, `motion` true, step 4000, `strips` `.cache/soccertrack_v2/ball_strips`.
  - `sources` {"stream": 21340}, `window_strips_allowed` false.
  - `n_train` 21,040 and `n_holdout` 300.
  - It also carries the label constants and recipe sha256s.
- Calibration is `<ckpt>.calib.json`, read by `load_calibration` (L367); `p_cal = sigmoid(a * logit + b)` (L374).
  - For v2.4000: a = 0.895072, b = -0.927959, tau (on p_cal) = 0.25.
  - It was fitted on the 4 M6 windows (st2_118578_*), 3,568 candidates.
- `check_unseen(ckpt, match, allow_seen)` (L422) refuses any match listed in the provenance `train`, in either half, and refuses a checkpoint with no provenance.

**4. Detections and top-k** (`_peaks`, L188-220)
- `prob = sigmoid(logit)`. A cell is a local maximum when `prob >= max_pool2d(prob, 5, 1, 2)`, i.e. the maximum of its 5x5 cells (ties all kept), and `prob >= tau`.
- The cell centre is `u = x0 + (gx + 0.5) * 4` and `v = y0 + (gy + 0.5) * 4`. It is kept if `cv2.pointPolygonTest(outline) >= -margin_px` (10 px).
- Candidates are sorted by prob (highest first) and cut to `[:top_k]`. Each is refined to the prob-weighted centroid of its 3x3 cells (clipped at map edges).
- `detect_v2` returns rows `(u, v, p_raw, size_px, logit)` in full-frame pixels. `size_px = 0.22 * max(local_scale(ground))` (NaN without calibration).
- The defaults are not the same everywhere:

| Where | tau | top_k | Notes |
|---|---|---|---|
| `detect_v2` | 0.05 | 8 | |
| `eval_ball_detector.py` (`V2_TAU, TOP_K`, L228) | 0.05 | 8 | rows `[u, v, p_raw, size, logit]` |
| `detect_ball_heatmap.py --v2` | 0.05 (on p_raw) | 8 | default `--stride 2` |
| v1 `detect` | 0.15 | 4 | |
| v1 in eval | 0.01 | 8 | |
| frozen tracker | 0.25 (on p_cal) | 4 | |

- The eval measures use the top-1. `parquet_candidates` (L480) re-sorts by column 2 and keeps 8.

**5. How the scripts run it**
- `eval_ball_detector.decode` (L398-477):
  - Pass 1 grabs frames and keeps the grey pitch-box crops at the needed sample frames.
  - Pass 2 reads sequentially with cv2 into a `RingBuffer(5)` and runs at t once t+2 is decoded. The tail frames are run without t+2.
  - `cache.drop_before(first + t - 350 - 50)`.
  - `v2_detectors` (L1280-1313) runs `check_unseen` against every window's match, then `load_v2`, then wraps `detect_v2(..., motion_zero=rgb_only, sigma_d, sigma_f)`.
- `detect_ball_heatmap.py --v2`:
  - Uses the same decode (`background_pass` L79, `dense_v2` L101-163).
  - Frames come from `v2_frames` (L69), plus `--extra-frames measured`.
  - Output rows are `V2_COLUMNS` (px, py, crop_h = size, confidence = p_cal, logit, p_raw, n_bg).
  - It writes a `<out>.meta.json` sidecar with sha256s.
  - `main_v2` (L217) refuses the test matches by window name and by `info["match"]`, requires a calib file unless `--no-calib`, and runs `check_unseen`.

**6. Training split sets** (L64-67)
- `V2_TRAIN = {"118575": (1, 2), "118576": (1, 2), "118577": (1, 2), "128058": (1, 2)}`. These are M3, M4, M5, M8.
- `V2_EXCLUDE = {("118576", 2): (13500, 20999), ("128058", 2): (13500, 20999)}`. These are the old T windows f015000 ±60 s. The test is `build_ball_strips.excluded` (L526): t±2 touching the range.
- `NEVER = {"118578", "117093", "132877", "117092", "128057", "132831"}`: M6 (U), the V-old pair (M2/M10), M1 (another camera) and the two test matches.
- These are enforced in three places:
  - `write_provenance` (L397) refuses a NEVER match in `train`.
  - `load_strips` (train L268-314, guard at L291-296) refuses strips outside `V2_TRAIN` or inside `V2_EXCLUDE`.
  - `build_ball_strips.check_allowed` (L532) refuses cutting them.

**7. Stream versus window ties (what a v3 has to change)**
- v2 strips were cut only from the source stream: `stream_frames` (PyAV over the signed remote half, L650), absolute frames, background over the whole half, `source = "stream"`.
- Window strips (`--window`, `window_frames` L670, OpenCV over the x264-re-encoded `data/.../<window>.mp4`, absolute = start_frame + local, background inside the window) are dropped by `load_strips` (L297-311) unless `--allow-window-strips`.
- `check_source` (L546) forbids mixing stream and window strips in one directory, because they share file names `<kind>_<match>_<half>_<frame>_<x0>_<y0>.jpg`.
- SOCCERTRACK_V2.md L703-705 says window motion input matches the strips (D std ratio 0.97).
- Frames are 4096x1080 in both sources, so window (u, v) are the same pixels as the stream. Tiles map to cells as `((u - x0) // 4, (v - y0) // 4)`; the 24 px box is ±3 cells.
- `A/tiles.parquet` has `abs_frame`. `B/auto_labels_AB.parquet` has only local `frame`, so start_frame has to come from the window json.
- **There are 18 verified windows, not 15:** 4 in A (M6) and 14 in B.
  - 12 are in NEVER matches (118578 x4, 117093 x4, 132877 x4).
  - 2 are the T windows (118576_2nd_f015000, 128058_2nd_f015000): v2's training match, but inside V2_EXCLUDE, with 0 v2 strips.
  - 4 overlap v2's training frames directly. Counts of v2 stream strips inside each window's frame range: 118575_1st_f015000 212, 118577_1st_f030000 190, 118576_1st_f030000 120, 128058_1st_f030000 80.
- **Consequences for a v3:**
  - `V2_TRAIN`, `NEVER` and the `write_provenance` and `load_strips` guards must be replaced (e.g. a v3 set that keeps M7/M9 and possibly M1 in NEVER).
  - Window strips need `--allow-window-strips`, or a re-cut from the stream at `abs_frame`.
  - After training, `check_unseen` will refuse v3 on all 18 windows. v2 is already refused on the 118575/118576/118577/128058 windows. A v2 vs v3 comparison on seen windows therefore needs `--allow-seen` (or leave-one-match-out). The clean comparison is M7/M9, where both are unseen.
  - The v2 calibration was fitted on M6, which becomes v3 training data.

**8. Runtime**
- No per-frame CPU figure for inference is documented. Documented figures:
  - The fused stem costs about 1.06x v1's full-frame forward (L103).
  - `median_u8` takes 0.07 s for 15x1000x4096.
  - Training took 84 min for 4,000 steps on 4 CPUs.
- I timed one forward here: `load_v2` on `.cache/ball_v2.4000.pt`, 1x6x664x3896 input, 4 threads, gives output (1, 166, 974) in 0.85-1.0 s after a 4.2 s first call. That excludes decode, motion planes and peak extraction.

# read:train

Report on train_ball_heatmap.py in v2 mode (--strips), for training a v3 from verified point labels

**1. Strip data the trainer reads (`load_strips`, L268-314)**
- The strips directory holds three files: `labels.parquet`, `labels_codes.npy` (opened memory-mapped, int8, shape (N, 80, 80), row-aligned with the parquet; L286 checks the shape) and an optional `labels_meta.json` (`{"constants", "coco", "seed", ...}`).
- `labels.parquet` columns, as found in the current `.cache/soccertrack_v2/ball_strips` (21,340 rows): file, match, half, frame (absolute frame in the half), kind, category, label, x0, y0, positive_only, weight, n_bg, has_prev, has_next, narrow, coco_balls, source, holdout.
- Each strip is one JPEG, 1280×320, made of four 320 px crops: [t-2 | t | t+2 | background, grey]. `label_ball_strips.split` (L457) cuts it up. `read_strip` (L339) treats prev, next and background as missing when has_prev is False, has_next is False or `n_bg < BG_MIN = 8`. OFFSET = 2.
- Codes, one per 4×4 cell of the t crop (SIZE 320, STRIDE 4, CELLS 80):
  - 1: inside a bag
  - 0: ignored
  - -1: negative
  - -3: hard negative, weighted ×3
- The current data has only `source == "stream"` and the matches 118575, 118576, 118577 and 128058.
- Holdout is the boolean column `holdout`. `label_ball_strips.holdout_mask` (L465) draws 300 strips, at most 1.5%, stratified by kind. `Batches` samples only from rows with holdout False (L702-703), and `monitor` scores the held-out rows every 250 steps.

**2. Loss (`strip_losses`, L493-523)**
- The network outputs logits of shape (B, 80, 80) (`build`, `head(...)[:, 0]`).
- **Bag term.** `has = (codes==1).sum > 0`. For a flight strip: r = `POOL_R` = 8 and off = log(`FOOTPRINT`=4). For any other strip: r = `LME_R` = 5 and off = log(n_bag). Then `z = (logsumexp(r·logit over the bag) − off)/r` and `pos = BCE(z, 1)·weight`. Whether a strip counts as flight comes only from `kind == "flight"`, at `Run.__init__` L740 and in `monitor` at L585.
- **Negative term.** It applies to cells coded -1 or -3, except on positive_only strips: `neg = Σ BCE(logit, 0)·σ(logit)^2·w`, with w = 3 (`HARD`) on -3 cells and 1 on -1 cells. `GAMMA` = 2.
- **Peak suppression.** Among the negative cells that are 5×5 local maxima (max_pool2d on detached logits), the 16 highest logits each add `BCE(·, 0)/16`.
- **Total** (`Run.step` L776): `(Σ_keep pos + Σ neg + Σ peak) / 16`. `keep` = `has` minus `drop_worst` (L526); from step `DROP_FROM` = 1200 it drops the worst 25% of flight bags, 15% of action bags and 25% of dribble bags (`DROP_V2`), with the count rounded at random.
- How each label type is encoded:
  - Positive bag: cells coded 1.
  - Negative cell: -1.
  - Hard negative: -3.
  - Ignore: 0.
  - A dribble strip is positive_only with weight 0.5, so it has no negative or peak terms.
  - A strip without a bag gets pos = 0 but still gets the negative and peak terms.

**3. Batches**
- Each batch is 16 strips (`MIX_V2`, L231): flight 5, action 3, dribble 1, lofted 1, and free 6 (landmark 2, cluster 2, boundary 1, random 1).
- `Sampler` (L317) draws uniformly with replacement within each pool. A free category with no strips falls back to all free strips. Any other kind with no strips stops the run: `SystemExit("no {k} strips to train on")`.
- `Batches` (L690) prepares the next batch in a background thread. Each strip gets its own generator seed (`default_rng(seed)`).
- `make_batch` (L468) returns x as float32 (16, 6, 320, 320) (RGB through `RGB_LUT`, then `motion_planes`) and codes as int64 (16, 80, 80).

**4. Augmentations (`AUG` L245, `draw_aug` L357, `augment_strip` L386)**
- All four crops get the same:
  - flip, p 0.5 (the codes are flipped too)
  - contrast 0.75-1.25 and brightness 0.75-1.25
  - saturation 0.8-1.2, as one luminance-keeping affine colour map
  - scale 0.8-1.25 in whole cells, with codes taken from the nearest cell; when scaling up, `_span` keeps the 1-cells inside the crop; when scaling down, it pads by reflection with code 0
  - blur, p 0.2, σ 0.3-1.0
- Each crop is then JPEG re-encoded on its own (p 0.3, quality 80-95).
- Each of t-2, t and t+2 gets its own luminance noise (σ 0-2).
- Time is reversed with p 0.5 and the motion is dropped with p 0.1.

**5. Optimiser and schedule**
- `optimiser` (L542): AdamW, weight decay 1e-4. Learning rates: head and lat 3e-4, `conv1_motion` 3e-4, layer2 1e-4, "early" (stem[0], stem[1], layer1) 3e-5. The early group is frozen until `FREEZE_TO` = 800.
- `lr_scale` (L562): warm-up over 100 steps, then cosine to 0.
- `STEPS_V2` = 4000; `torch.set_num_threads(4)`. Seed 0 by default (`--seed`).
- Monitor (L566): bag hit, peakiness, negative peaks with p > 0.15, and the norm of the motion weights. From step 1500 the run aborts with exit status 3 if peakiness exceeds 30 (`PEAKY_MAX`).

**6. Checkpoints and provenance**
- Checkpoints are saved at `CHECKPOINTS_V2 = (2500, 3250, 4000)` as `<out>.<step>.pt`.
- Each gets `<ckpt>.provenance.json` through `ball_heatmap.write_provenance(ckpt, provenance_info(...))` (L651). It records:
  - train ({match: halves} taken from the table) and exclude (`V2_EXCLUDE`)
  - label constants, `label_constants_as_code`, `labels_coco`, `labels_seed`
  - sigma `{SIGMA_D 0.0535, SIGMA_F 0.0174}` and `sigma_measured`
  - `recipe_sha256` (hash of `train_ball_heatmap.py` and `label_ball_strips.py`) and `files_sha256`
  - seed, motion, step and steps
  - strips dir, `window_strips_allowed`, `labels_sha256`
  - `n_train` and `n_holdout` by kind, sources
  - mix, lr, drop and loss constants, `AUG`, and the monitor history
- `load_v2` reads the sigmas back from the provenance at inference.

**7. Guards (exact)**
- `train_ball_heatmap.load_strips` L291-296 stops with a SystemExit, and **no flag overrides it**, for any row whose match is in `NEVER = {118578, 117093, 132877, 117092, 128057, 132831}`, whose half is not in `V2_TRAIN` (118575, 118576, 118577, 128058, halves 1 and 2), or where `build_ball_strips.excluded(f, m, h)` is true (t±2 touches `V2_EXCLUDE = {(118576,2): (13500,20999), (128058,2): (13500,20999)}`; frame must be absolute).
- Same function, L297-311: rows with `source != "stream"` are dropped and counted, and a table with no `source` column is refused. The flag `--allow-window-strips` lifts both, and the provenance records `window_strips_allowed: True`.
- `check_sigma` (L785) refuses placeholder motion scales; `--placeholder-sigma` lifts it.
- `ball_heatmap.write_provenance` (L405-407) raises ValueError if any NEVER match is in "train". No flag lifts it, and it would only fire at step 2500.
- Upstream:
  - `label_ball_strips.label` (L489) refuses halves outside `V2_TRAIN`; `--allow-any` lifts it.
  - `build_ball_strips.check_allowed` (L532) always refuses `TEST` matches; for a match outside V2_TRAIN it requires both `--window` and `--allow-any`.
  - `check_source` (L546) allows only one source per directory.
- At evaluation, `check_unseen` refuses to score a checkpoint on a match it was trained on; `--allow-seen` lifts it.
- **How this applies to the verified windows.** I read only window names and frame ranges. A has the 4 windows of 118578 (M6). B has 14 windows. That is 18 windows in total, not 15.
  - The 12 windows of 117093, 118578 and 132877 are in NEVER, so `load_strips` and `write_provenance` refuse them with no flag.
  - `st2_118576_2nd_f015000` and `st2_128058_2nd_f015000` lie inside `V2_EXCLUDE`, so they are refused.
  - The other 4 windows (`118575_1st_f015000`, `118576_1st_f030000`, `118577_1st_f030000`, `128058_1st_f030000`) pass the match guard. They would still be dropped as window strips unless `--allow-window-strips` is given.
  - B frames are local to the window: absolute frame = window start + frame. A already has `abs_frame`.
- So v3 needs a new allowed set (for example a `V3_TRAIN` and a narrower `NEVER` in `src/ball_heatmap.py`), and `load_strips` must accept it. That is a protocol change; it should not be made by passing flags.

**8. Run time**
- `research/session_scratchpad/ball/train_v2.log`: the motion model took 5043 s for 4000 steps, about 1.26 s per step on 4 CPUs including the monitor (about 1.14 s per step in the first 100 steps). `SOCCERTRACK_V2.md` says "4,000 steps, 84 min on 4 CPUs".
- `train_v2rgb.log` (RGB-only): 4239 s, about 1.06 s per step.
- I found no logged `--bench` output.

**9. Minimal change to train from point labels**
- **Where the targets are built.** `label_ball_strips.label` (L485) runs `flight_codes`, `action_codes`, `dribble_codes`, `lofted_codes` or `free_codes` on each row of `strips.parquet` and writes the three files. The trainer reads only those files, so a `verified_codes(row, tiles)` function (or a separate labeller) writing the same format is enough. The strips themselves can be cut with `build_ball_strips.cut(targets, window_frames(path, start, first, last), ...)`. Each target needs kind, absolute frame, x0 and y0.
- **Does a small bag around a verified ball plus -3 cells at N tiles fit the loss unchanged? Yes.**
  - `strip_losses` makes no assumption about bag shape.
  - Bag: the cells whose centre `(x0+4j+2, y0+4i+2)` falls in the 24 px box around (u, v), about 6×6 to 7×7 cells. Write the 1-cells last, as `action_codes` does, so they win over overlapping N tiles.
  - At N tiles: -3 on a small disk, which then also feeds peak suppression.
  - At B* and S tiles: 0, the same way spare balls get COCO ignore disks.
  - Elsewhere: -1 on strips from frames with a verified B (the match ball is located). On frames with N tiles only, use 0 except at the N tiles.
- **Changes the trainer does need:**
  1. **Pooling.** Use footprint pooling for the new bags, by setting the flight mask to `np.isin(kinds, ("flight", "verified"))` at L740 and L585. Under log-mean-exp, one peak cell in a 49-cell bag scores z−0.78 and a uniform glow scores z. Under footprint pooling the peak scores z−0.17 and the glow z+0.31. A new kind name also escapes `DROP_V2` automatically, so verified bags are not dropped as hard cases.
  2. **Mix.** Pass a new mix, for example `(("verified", None, 10), ("free", None, 6))`, because `Sampler` stops the run if a flight, action, dribble or lofted pool is empty. `train_v2(mix=...)` exists, but no command-line flag sets it.
  3. **Data guard.** Give `load_strips` and `write_provenance` the new allowed match set, plus the window source.
  4. **Provenance.** Add the new labeller to `recipe()`.
  5. **Holdout.** Set it by window (stratified holdout puts neighbouring frames of the same window on both sides).
  6. **Peakiness abort.** It cannot trigger on bags of 49 cells or fewer, because a bag can never have more than 30 cells above 0.5 unless it is larger than that.

# read:strips

**Strip pipeline (build_ball_strips.py, label_ball_strips.py): code-reading report**

**1. How a strip is cut**

- **Picture.** One JPEG, 1280×320 BGR, written by `cv2.imwrite` at q98 (`QUALITY=98`, `SIZE=320`). It is `[t-2 | t | t+2 | B_t]`, built by `assemble()` (L688-695). All four crops come from the same full-frame place `image[y0:y0+320, x0:x0+320]`. `OFFSET=2` comes from `src/motion_bg.py`.
- **Background B_t.** Built by `motion_bg.BackgroundCache(first, last, crop=box)`. It is the per-pixel median (`median_u8`) of the grey frames that are multiples of `BG_STEP=50`, within `BG_HALF_SPAN=350` of `c = bg_centre(t)` (the nearest multiple of 50) and inside `[first, last]`. With fewer than `BG_MIN=8` samples there is no background. It is stored grey as three equal channels (`GRAY2BGR`). The crop is `bg[y0-by0 : y0-by0+320, x0-bx0 : x0-bx0+320]`, where `box = ball_heatmap.pitch_box((H, W), poly)`: the outline's bounding box, 32 px wider on each side, 64 px higher and 32 px lower.
- **Missing pieces.** A missing t-2 or t+2 is stored as t. A missing background is stored as grey(t). The flags `has_prev`, `has_next` and `n_bg` record which (`cut()` L731). `read_strip` (train L339-354) turns these into None, i.e. zero motion.
- **Decode loop.** `cut(targets, frames, first, last, box, out, match, half, write=None, ...)` (L698-762) sorts targets by frame and decodes only the frames that are multiples of 50 or a wanted t-2/t/t+2. A strip is written once `cache.ready(t)`. It returns the targets with `file`, `n_bg`, `has_prev`, `has_next` added.
- **Two frame sources:**
  - **Stream (v2 training):** `stream_frames(open_half(match, half), k0, last)` (L638-667) decodes the source panorama with PyAV, frame index from pts. The background spans the whole half (`first=0, last=1e9`).
  - **Window:** `window_frames(path, start, first, last)` (L670-681) reads with `cv2.VideoCapture` sequentially; absolute frame = `start_frame` + local index. The background is limited to the window (`first=start`, `last=start+frames-1`).
- **Coordinates.** Window mp4 pixels and source-stream pixels are the same, with no offset or scale. `st.write_clip` (src/soccertrack_v2.py L507-534) re-encodes the stream at full resolution: libx264, crf 18, veryfast, yuv420p. The JSON gives the width and height (4096×1080; 132877 is 4096×1084). The only offset is in time: absolute = `start_frame` + local.
  - In `A/tiles.parquet`, `frame` is local and `abs_frame = start + frame` is also given. `B/auto_labels_AB.parquet` and `B/judged_*.parquet` have the local `frame` only.
- **Pixel differences between the two sources.** A window has an extra x264 generation and OpenCV colour conversion instead of PyAV's. Its background is shorter at the window edges: for the last ~25 frames n_bg is 7, so there is no B. `label_ball_strips.compare_window` / `window_values` (L673-839) measure this gap.

**2. On-disk format (output directory)**

- **Strip files:** `<kind>_<match>_<half>_<frame:06d>_<x0>_<y0>.jpg`, named by `strip_name()` (L684). Strips with the same kind, frame and place are deduplicated (main L918-921).
- **`strips.parquet`:** frame (absolute, int64), kind, category, label, x0, y0, a_x, a_y, b_x, b_y, s, t_hat, L, event_a, event_b, player_a, player_b, file, n_bg (int64), has_prev, has_next (bool), match (str), half, width, height, source ("stream" or the window name), allow_any.
- **Written by:** `resume_cut` (L796-833), which saves every 200 strips to `strips_<m>_<h>_<source>_<first>-<last>.partial.parquet`, then `merge(out, rows, match, half, first, last, kinds, source)` (L836-852, with a file lock).
- **Labeller outputs:**
  - `labels.parquet`: file, match, half, frame, kind, category, label, x0, y0, positive_only, weight, n_bg, has_prev, has_next, narrow, coco_balls, source, holdout.
  - `labels_codes.npy`: (N, 80, 80) int8, row-aligned with `labels.parquet`.
  - `labels_meta.json`: `{constants: LABEL_CONSTANTS, coco, seed, strips, labelled, holdout}`.
- **Current v2 directory:** `.cache/soccertrack_v2/ball_strips` has 21,576 strips, all `source == "stream"`, and 21,340 labelled.

**3. Codes per 4×4 cell (label_ball_strips.py)**

Cell (i, j) of the t crop has its centre at `(x0 + (j+0.5)·4, y0 + (i+0.5)·4)`. `Canvas` (L151-215) maps each centre to the ground with `st.image_to_pitch` and the match calibration (K, D, Knew, Hinv), and gets the local px/m from `scales()`.

| Kind | Code 1 (the bag) | Code 0 (ignore) | Codes -1 / -3 |
|---|---|---|---|
| `flight_codes` (L274-319) | Cells projecting into `bag_window(t_hat, L)` on A→B, within 2.5 m of the line or 20 px of its image, plus up to 1 m above | Bag cells on a body or within 1 m of feet; an ignore band of 6 m / 48 px of the whole line plus a 2 m curtain | -3: bodies ≥1 m from the bag and landmarks (0.75 m, at least 12 px) outside the band. Rest -1. The strip is kept only if 85% of its bag (on a 640 px canvas) is inside the crop, else retried narrower, else dropped |
| `lofted_codes` (L322-352) | None | Curtain up to `clip(0.3 L, 3, 15)` m above the line, plus within 4 m / 32 px of it, and players there | -3: other players. Rest -1 |
| `action_codes` (L372-401) | v1 region 1.5 m and 0.5 m up around actor A and track point T (THROW IN: 2.6 m up) | 0.5 m band around it | -3: other players' bodies and the actor's head |
| `dribble_codes` (L404-413) | 1.5 m around the dribbler | Everything else | None: `positive_only`, weight 0.5 |
| `free_codes` (L427-454) | None | With `--coco`: 1 m disk around a COCO ball (yolov8s, conf 0.25, crop upsampled ×2) away from the landmarks | Start at -1; `body_slices` (1.9 m up, 0.3 m+10 px to each side) set to -3; `landmark_mask` set to -3 |

Inputs to the codes: GSR feet (`People`), BAS events, the release ball track (`.cache/soccertrack_v2/ball/*.npz`, including its status) and the calibration. All are cached for the non-sealed matches.

**4. How free strips are chosen (`free_samples`, L472-500)**

- **Frames:** action frames (deduplicated by frame) where the actor is known, the track point exists with status ≠ 0, and the actor is within 3 m of it. Every second such frame is used, with 2 strips each.
- **Category:** drawn by `FREE_MIX` = cluster 0.35, landmark 0.25, boundary 0.25, random 0.15.
- **Centre (`free_centre`, L436-469):**
  - cluster: a player with at least 2 others within 5 m;
  - landmark: one of the 29 `LANDMARKS`;
  - boundary: a touch or goal line picked with weights 105:105:68:68, then 50% up to 2 m inside or 50% up to 60 px outside along the image normal;
  - random: uniform in the outline polygon.
- **Placement:** `place()`, then `crop_ground()` must give every sampled cell ground point at least 12 m from the actor and the track point. Up to 10 tries.
- **`place(bbox, rng, limits)` (L204-213):** centres the crop on the bag box with jitter ±min(100, (320 − extent)/2 − 8) per axis, clipped into `limits = box`.

**5. How window strips are flagged and guarded**

- `source` = the window name, `allow_any` = True.
- `check_allowed` (L532): refuses `TEST = {128057, 132831}` always; a non-`V2_TRAIN` half needs `--window --allow-any`.
- `check_source` (L546): stream and window strips never share a directory.
- `label()` (L485): refuses non-`V2_TRAIN` halves unless `allow_any`; builds `Half(match, half, window=source)`.
- **Trainer `load_strips` (train L268-314):**
  - refuses `NEVER` matches, halves not in `V2_TRAIN`, and frames where `excluded()` is true (`V2_EXCLUDE`);
  - drops `source != "stream"` unless `--allow-window-strips`.

**6. Calls a new v3 cutter needs, per window w**

```python
info = json.load(DATA_DIR/"soccertrack_v2"/f"{w}.json")
match, half, start = str(info["match"]), int(info["half"]), int(info["start_frame"])
first, last = start, start + int(info["frames"]) - 1
W, H = info["width"], info["height"]
poly = np.asarray(bs.outline(match, half, W, H), np.float32)   # = info["pitch"]
box  = bs.pitch_box((H, W), poly)
x0, y0 = bs.place((u, v, u, v), rng, box)       # or own choice; MUST satisfy
#   box[0] <= x0 <= box[2]-320 and box[1] <= y0 <= box[3]-320
#   (cut() does not check: a bg slice off the box breaks np.hstack)
targets = [bs._row("verified", start + f_local, x0, y0, category=..., label=...)]
bs.check_source(out, w)
rows, _ = bs.resume_cut(out, targets,
        lambda k0: bs.window_frames(DATA_DIR/"soccertrack_v2"/f"{w}.mp4", start, k0, last),
        first, last, box, match, half,
        extra=dict(match=match, half=half, width=W, height=H, source=w, allow_any=True))
bs.merge(out, rows, match, half, first, last, kinds=("verified",), source=w)
bs.partial_path(out, match, half, w, first, last).unlink(missing_ok=True)
```

- Group tiles by frame so that one crop carries several tiles. The tile's cell is `((v - y0)//4, (u - x0)//4)`.
- The CLI `--like` route is not suitable: `--kinds` only accepts `KINDS`, and main() drops targets that `excluded()` flags.
- The labeller must write `labels.parquet`, `labels_codes.npy` and `labels_meta.json` as in section 2. The trainer reads file, match, half, frame, kind, category, positive_only, weight, n_bg, has_prev, has_next, source, holdout.

**Changes the trainer needs before v3 can run**

- **Guards in `load_strips`.** Of the windows in the verified files, only st2_118575_1st_f015000, st2_118576_1st_f030000, st2_118577_1st_f030000 and st2_128058_1st_f030000 pass its guards:
  - 118578 (all of A), 117093 and 132877 are in `NEVER`.
  - st2_118576_2nd_f015000 and st2_128058_2nd_f015000 (frames 15000-19499) lie entirely inside `V2_EXCLUDE` (13500-20999).
  - All of them are window strips, so they are dropped unless `--allow-window-strips`.
- **Batch mix.** `Sampler` draws only the kinds in `MIX_V2` (L231), so a new kind needs a mix entry.
- **Loss pooling.** `strip_losses` uses the footprint-4 smooth max only where kind == "flight", and log-mean-exp over the bag otherwise.
- **Motion scales.** For comparability with v2, keep `SIGMA_D=0.0535` and `SIGMA_F=0.0174`, which were measured on stream strips.

I found 18 windows in these files, not 15: 4 of 118578 in A and 14 in B. Sealed matches were never shown: the one cache listing that could hold their files was filtered by name before printing.

# read:eval

**Evaluation subsystem for v3: window splits, `--save` format, scoring against verified labels, and a held-out design that leaves M7/M9 untouched**

**1. Windows and matches**

Match ids are in `src/soccertrack_v2.py:71-77` (`MATCHES`). The sealed set is `TEST_MATCHES = {"128057","132831"}`, defined in `eval_ball_detector.py:229` and `build_ball_verify.py`. `window_data()` refuses those matches at line 387. To respect the seal I searched only by the ids of the other matches, so I don't know whether any M7 or M9 windows exist on disk.

The verified labels cover **18 windows, not 15**: 4 for M6 in `A/` plus the 14 others in `B/`. The task's count of 15 looks wrong. M1 has a window, `st2_117092_1st_f030000`, but it has no labels and is never used for training (other camera).

| role | windows | match |
|---|---|---|
| **U, the tuning match** (`calibrate_ball_heatmap.U_MATCHES={"118578"}`); never trained on (`ball_heatmap.NEVER`, line 67) | `st2_118578_1st_f030000`, `_1st_f052500`, `_2nd_f015000`, `_2nd_f037500` | M6 |
|   U-E (events tuning, `sweep_events_v2.py:80`) | `1st_f030000`, `2nd_f015000` | M6 |
|   U-test (tracker gate) | `1st_f052500`, `2nd_f037500` | M6 |
| **Old T** (diagnostic only; v2 kept these ±60 s out via `V2_EXCLUDE`) | `st2_118576_2nd_f015000`, `st2_118578_2nd_f015000`, `st2_128058_2nd_f015000` | M4, M6, M8 |
| **v2 training halves** (`V2_TRAIN`, lines 64-66) | `st2_118575_1st_f015000`, `st2_118576_1st_f030000`, `st2_118577_1st_f030000`, `st2_128058_1st_f030000`, plus the M4 and M8 T windows above | M3, M4, M5, M8 |
| **V-old** | `st2_117093_1st_f015000`, `st2_132877_1st_f015000` | M2, M10 |
| **V-new** (already used up by protocol v4) | `117093_{1st_f045000, 2nd_f015000, 2nd_f045000}`, `132877_{1st_f045000, 2nd_f015000, 2nd_f045000}` | M2, M10 |
| **Sealed** | none were opened | M7 128057, M9 132831 |

**2. What `--save` writes, and the `cand_*.json` files**

The top-level keys are `{"summary","detectors","protocol","results"}`; the full schema is in the docstring at `eval_ball_detector.py:120-190`. I checked one file, `v5/cand_all_st2_128058_1st_f030000_placebo.json`:

- **`results`:** one entry per window, with:
  - `window`, `match`, `half`, `start_frame`;
  - `frames`: the full frame sets of the window (passes with `frames` = 4 local frames, `s`, `from`/`to` in pitch metres, `kick`, `reception`, `block`; plus `actions`, `random` (50), `placebo_pre`, `placebo_post`);
  - `people`, `n_bg`, `frames to measure`;
  - one entry per detector with `candidates: {"<local frame>": rows}`, best first, at most 8 (`TOP_K`).
- **Row formats** (u, v in window pixels):
  - v2: `[u, v, p_raw, size, logit]`, with `p_raw` ≥ `V2_TAU` 0.05;
  - v1: `[u, v, p, size]`;
  - COCO: `[u, v, conf, box_h]`.
- **`detectors`:** `kind`, `path`, `calib`, `train` provenance, `ckpt sha256` (v2 = `31f9a5eb...`), `sigma`.
- **One job per window and frame kind.** `--frames {passes,placebo,actions,random}` (`FRAME_KINDS`, line 287; `measured_frames()`, line 290). Each `cand_*` file therefore has candidates only at that kind's frames: 9 in the file I checked.
- **Don't read the per-file `summary`.** The legacy measures count unmeasured frames as misses. Merge the files with `ev.load_scores(paths)` (line 1077) or `--reuse`.
- **Which detectors each file holds** (`v5/cand_queue.sh`):
  - `cand_all_*`: COCO + v1 + v2, run with `--allow-seen`, on the 6 training-match windows;
  - `cand_v1_*`: v1 only, on V-new;
  - `cand_v2_*`: v2 only, on V-old.
- **The rest of the candidates are elsewhere:**
  - M6: `ball/U_a.json` and `U_b.json` (sealed; COCO, v1, v2 at 2500/3250/4000);
  - V-old: `V.json`;
  - V-new: `Vnew_sealed.json` (COCO, v2, T).

**The verified labels sit on exactly these frames.** `A/frames.parquet` per window is 117 + 192 + 218 + 235 = 762 frames, the same as the eval's "frames measured of" counts. Frames are local, with `abs_frame = start_frame + frame`, and u, v are in the same pixel coordinates as the candidates.

**3. Scoring a top-1 pick against verified (u, v)**

No existing function does this. The pieces exist:

- **Candidates:** `ev.load_scores([...])` gives `merged[w][name]["candidates"][str(f)]`.
- **M6 truth:** `A/frames.parquet` with `status=="ball"` and its `u, v`. That is 546 frames. The other statuses are `unknown` 213, `conflict` 3 and `decoy` 28. The per-frame rule is `label_ball_verify.frame_status()` (line 98, `SAME_BALL_PX = 24`).
- **B truth:** from `auto_labels_AB.parquet`, take the frame's top-`p` tile with `auto=="B"`, and let the `judged_*.parquet` rows override it. That gives 1,927 of 3,048 frames. 590 frames have more than one B tile; merge those like `frame_status`.
- **Hit rule:** `rows[0]` with `|u0-u| ≤ 12` and `|v0-v| ≤ 12`, i.e. the ball centre inside the top-1's 24 px box, which is the judges' criterion. A missing frame key means not measured and is left out; an empty list is a miss. `validate_ff_auto.CONFIRM_PX = 12` (line 70) is the Euclidean version used for "confirmed false hits" (lines 334-343).
- **Statistics to reuse:**
  - `validate_ff_auto.agreement` (102), `mcnemar` (117), `mcnemar_one_sided` (136), `binom_upper` (126);
  - a block bootstrap like `ev.compare_ff` (line 1150), with key `f"{w}:{f//1500}"`.
- **Unknown frames:** a top-1 within `MERGE_PX = 6` px of an N/S/B* tile is a confirmed false pick; anything else is unverifiable.
- **Bias:** the tiles came from v2 top-2, v1, COCO, T and the pass line. On M6, v2's top-2 is among 0.888 of the verified balls (`A/summary.json`). Verified-ball frames therefore favour v2-like detectors, and v3 hits on balls nobody proposed land in `unknown` and cannot be scored.

**4. Eval run time** (CPU, `--threads 4`, from the logs)

- **Per window:**
  - background pass 30-60 s;
  - v2-architecture checkpoint about 1.65 s per frame (U_a: 316 s for 192 frames);
  - v1 about 1.6 s per frame; COCO about 2.8 s per frame.
- **Whole windows:**
  - one 3-minute window (190-290 frames) is about 6-9 min per v2-type checkpoint;
  - U_a (2 windows, 5 detectors) took 3,047 s; U_b took 3,979 s.
- **For v3 on all of M6** (762 frames): about 21 min per checkpoint plus about 4 min for decode and background. Three checkpoints in one decode come to about 70 min with `--no-coco`; reuse the v1, v2 and COCO candidates from `U_a.json`/`U_b.json`.

**5. How v2 was compared with v1** (protocol v3, `SOCCERTRACK_V2.md` 581-837)

- **Tuning set:** all choices on U = M6 (4 windows, 63 passes).
- **Primary measure, FF-auto** (`ff_hit`, line 364):
  - the top-1 at any p, at 4 frames per pass (`FF_S`);
  - within 4 m of the passer-receiver segment, |t-s| ≤ 0.30, and 1 m or more from every person.
  - Its agreement with the judge is kappa 0.82 on T and 0.81 on V-old (`validate_ff_auto.validate`).
- **Paired test:** `compare_ff`, a 1-minute block bootstrap with 4,000 draws, seed 0. `summarise()` pairs v2 with COCO and v2 with v1 automatically (lines 1248-1250).
- **Placebos:** before the kick (−15 and −30 frames) and after the reception (+15).
- **Legacy measures:** the 1 m free-flight hit and the action hit.
- **Checkpoint:** chosen by the best FF-auto on U among 2500/3250/4000, giving 0.603, 0.599 and **0.627**. The decision is recorded in `U_decision_motion.json`.
- **Result:** v2 minus v1 was +0.004 [−0.05, +0.06]. H2 (v2 better than v1) was never tested because gate G1 failed. The legacy measures favour v1 (action hit 0.34 against 0.43). The judged blind tiles were only ever used for v1 against COCO and for tracked against raw v2.

**6. Proposed held-out design for choosing v3's checkpoint**

1. **Train on B only:** the 14 windows of M2, M3, M4, M5, M8 and M10. Leave out M6 entirely (images and labels) and M1. Leave out M7 and M9 for good.
2. **Choose on M6 alone**, using the `A` labels: human-judged, two judges plus a third, kappa 0.935, 546 ball frames.
   - **Primary measure:** top-1 hit rate on M6 frames with `status=="ball"`, using the 12 px box rule. Every miss there is determined, so it needs no new judging.
   - **Rule:** best primary among the saved steps (`CHECKPOINTS_V2 = (2500,3250,4000)`, `train_ball_heatmap.py:235`), taking the later step when within 0.02, as in protocol v1. Use all 4 windows: a standard error of about 0.02 on 546 frames is needed to tell checkpoints apart, and splitting U-E from U-test would halve that power for a choice that makes no claim.
   - **Secondary and guards:**
     - FF-auto and the placebos through `--reuse U_a.json U_b.json v3.json`;
     - the confirmed-false rate on `unknown` frames;
     - pre-kick placebo no higher than v2's.
   - Then fit Platt and `tau` on M6, in the registered order.
3. **Read v3 against v2 on M6 as a diagnostic only.** v2's checkpoint, Platt fit and `tau` were also chosen on M6, and the frame set is biased towards v2 (section 3). The claim belongs on M7/M9. That needs a registered unlock of `TEST_MATCHES`, and fresh blind tiles with v3 added to `build_ball_verify.SOURCES`.
4. **Disclose:** the B auto-labels come from a tile classifier trained on M6 plus 500 judged tiles. That carries M6 judgements into v3's training labels, though not M6 images. Without M6 the classifier's precision was 0.55, so no clean substitute exists.

**Code changes needed before this works:**

- **Provenance refusal:** `ball_heatmap.NEVER` contains 117093 and 132877 (M2, M10), so `write_provenance` would refuse v3's training set. The strip checks against `V2_TRAIN` (`train_ball_heatmap.py:271-296`, `label_ball_strips`, `build_ball_strips`) would also refuse it. v3 needs its own `V3_TRAIN`/never-set, still containing 118578, 117092, 128057 and 132831.
- **Naming and pairing:** name the checkpoints `ball_v3.<step>.pt`, since the detector name is `Path(path).name`. v3 will be of kind "v2", and `summarise()` does not pair two v2-kind detectors, so call `ev.compare_ff(results, "ball_v3.N.pt", "ball_v2.4000.pt")` explicitly.
- **Unseen check:** `check_unseen` passes on M6 without `--allow-seen` only if v3's provenance `train` leaves out 118578.