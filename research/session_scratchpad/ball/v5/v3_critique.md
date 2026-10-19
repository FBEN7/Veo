

# critic:ml

**Ranked issues: ML soundness of the v3 plan (`v3_plan.md`)**

Evidence comes from parquet checks on the B labels. `judged_spot2` is the only out-of-sample check of `auto_labels_AB`; the classifier was trained on `judged_train` and `judged_spot`, so its labels on those are in-sample.

### Blockers

**1. Weight-3 hard negatives (-3) on auto "not" tiles with p ≤ 0.1.**
- In spot2, 10 of the 50 auto "not" tiles are balls: 3 B, 1 B*, 6 S.
- All 3 missed match balls are in unlocated frames (3 of 34 there, plus 4 S).
- This is expected: in an unlocated frame, the true ball, if any candidate found it, can only carry "not" or "judge".
- The plan centres the unlocated-frame negative strips on exactly these tiles. About 9% of those -3 targets are the match ball, and they get ×3 weight plus peak suppression.
- Fix:
  - Use -3 only on judged N tiles (268 of them).
  - Code auto "not" as -1, and only in located frames, where the ball is known to be elsewhere.
  - In unlocated frames, centre negative strips only on judged N tiles, or drop them.

**2. Spare balls are labelled inconsistently.**
- Judged S / B* get 0, but unjudged spare balls get -1 or -3. M6 had 450 S/B* tiles among 2,403 (19%), so they are common.
- S also covers the match ball when it is out of play, which looks identical in 3 frames.
- A static test separates them well: a tile counts as static if 2 or more other frames of the same window have a tile within 6 px. That holds for 100% of judged B*, 81% of S, 22% of N and 12% of B.
- 1,601 of the 4,548 auto "not" (p ≤ 0.1) tiles are static.
- Fix:
  - Put a 0-disk of radius 4 cells (16 px) on every static non-B tile not judged N, and on every COCO-sourced one (62% of S are COCO, 13% of N).
  - The plan's 2-cell disk is too small, because the network's blob spills onto the -1 cells around it.
  - Also put 0 on the unjudged "judge" tier (618 tiles; 77% of an earlier judged sample of "judge" tiles were B) and on auto "not" with p > 0.1. These are "not" only because they were away from the top tile.

### Major

**3. No ignore ring around the bag.**
- -1 starts 12 px from a noisy centre: the mean of the candidate generators' positions.
- About half the measured frames are action frames (2,164 tiles), where the ball sits at a player's feet. Peak suppression puts its gradient on exactly these cells.
- Fix: code 1 for |du|, |dv| ≤ 12 px; code 0 for ≤ 24 px; -1 beyond.

**4. Too many epochs on near-duplicate data.**
- There are 1,927 located frames. Grouped by gaps of 15 frames they form 588 clusters; by gaps of 50, 259 clusters.
- 3,000 steps × 8 = 24k draws: about 14 epochs, or about 40 draws per cluster.
- Each frame has a single crop position, baked into the JPEG, so only flips and scaling vary between epochs.
- Fix:
  - Cut 3 placements per located frame, each with its own `place()` seed.
  - Train 2,000 steps, with checkpoints at 500 / 1000 / 1500 / 2000.
  - Keep an EMA copy of the weights (decay 0.999) and select among the EMA checkpoints. The 1000 checkpoint under cosine is still at about 75% of peak LR and noisy.

**5. The selection measure mostly measures re-ranking of v2's proposals.**
- 96% of B tiles include v2 as a source, and v2's top-2 covered 0.888 of M6's verified balls. Both training positives and selection frames are therefore v2-seeded.
- The guard counts a top-1 as false only near judged N / S / B* tiles. A v3 whose top-1s land where no tile was proposed counts as "unverifiable" and passes, so the guard favours divergence.
- Fix:
  - Report the unverifiable top-1 rate on the 213 unknown frames, and cap it at v2's + 0.05.
  - Before Platt, blind-judge the chosen checkpoint's top-1 on the unknown frames (about 213 tiles).
  - Split the primary by `sources_hit` (contains v2 or not).
  - Add hit@4 as a secondary measure, since the tracker uses the top 4.
  - 0.02 is roughly one paired standard error on 546 clustered frames. The tie rule is fine, but do not read anything smaller as a difference.

**6. The monitor will saturate.**
- In-crop bag-hit on 320 px crops is close to 1 by construction, so it cannot show overfitting.
- Fix: score full-frame top-1 on `st2_128058_2nd_f015000` (166 located frames) at each checkpoint, about 6-9 min each.

### Minor

7. **The v2 mix needs exact counts.** "MIX_V2 proportions" cannot be split into integers over 5 strips. Use flight 2, action 1, free 2 (cluster 1, landmark 1).
8. **Contradicting labels from v2 strips.** Drop the 602 v2 strips that fall inside B windows' frame ranges (212 + 190 + 120 + 80); their weak codes can contradict the verified labels on the same frames.
9. **Match imbalance.** M2 + M10 give 1,279 of 1,927 located frames (66%). M8 keeps only 60 frames (`st2_128058_1st_f030000`) once the monitor window is held out. Sample with weight ∝ √(frames per match).
10. **Batch norm.** Keep BN in eval mode for the stem and layer1 (learning rate 1e-5), so 16 ball-centred crops do not move the running statistics used on the full panorama.
11. **Window vs stream.** All verified strips are cut from the window mp4s and all v2 strips from the stream, so the source goes with the label type (windows are mostly positives). This is acceptable because evaluation also decodes windows. Check that the strips per source and kind are logged in the provenance.

### Right; do not change
- Fine-tune from v2 rather than train from scratch: about 260-590 effective clusters are too few for scratch training.
- Same architecture and the same SIGMA_D / SIGMA_F.
- Learning rates at about a third of v2's, with warm-up and cosine.
- Footprint pooling rather than log-mean-exp: the latter rewards a uniform glow over a peak.
- A new kind name, so the verified bags escape DROP_V2. B precision is about 96-100% (spot2: 50 of 50), so no drop-worst is needed.
- Judged tiles override auto labels, and frames whose B tiles are more than 24 px apart are dropped.
- -3 on judged N tiles, with the 1-cells written last so they win.
- M6 kept fully out of training and used for selection.
- The selection rule fixed before training.
- The monitor holdout is a whole window, and it sits inside V2_EXCLUDE, so v2 never trained on it.
- v2 strips kept to limit forgetting.
- The leakage through the tile classifier disclosed.

# critic:feasibility

**v3 plan: feasibility review in this codebase**

The plan is buildable, but not as written. Four blockers, all confirmed in the code, stop the trainer from expressing it. Counts below come from the B parquets, M6's 4 windows (U_a/U_b) and a short decode timing on one window.

**Blockers**

1. **The trainer cannot fine-tune or use the plan's recipe.**
   - `Run.__init__` (train L734+) builds a fresh net and loads nothing.
   - `optimiser()` hard-codes `LR_V2` and freezes the early layers until `FREEZE_TO` (800).
   - The CLI (L960) passes neither `checkpoints` nor `mix`. With `--steps 3000` you get only 2500 and 3000, not 1000/2000/3000.
   - Fix:
     - Add `init`, `lr` and `freeze_to` to `Run`, `optimiser` and `train_v2`. After `build`, call `load_state_dict(torch.load(init))` strictly, after checking v2.4000's sha256 (31f9a5eb…).
     - Add a `--v3` CLI mode that sets `MIX_V3`, `LR_V3`, `freeze_to=0`, `CHECKPOINTS_V3=(1000,2000,3000)`, 3000 steps and the init path.

2. **The guards refuse 10 of the 14 windows, and the crash comes late.**
   - `load_strips` (L291-296) refuses `NEVER` matches (117093 and 132877: 8 windows) and anything inside `V2_EXCLUDE`. That includes both T windows, one of which is the monitor holdout (128058_2nd). Holdout rows are refused too.
   - It also drops every window strip (L297-311).
   - `write_provenance` (bh L405) has no override and only fires at the first checkpoint, so the run would die after about 20 minutes of training.
   - Do not narrow the global `NEVER`: `ball_heatmap._check` L663 asserts that a 117093 training set is refused.
   - Fix:
     - Add `V3_WINDOWS` (the 13 training windows plus the holdout window), `NEVER_V3 = {118578, 117092, 128057, 132831}` and `write_provenance(ckpt, info, never=NEVER)`, with v3 passing `NEVER_V3`.
     - In `load_strips(protocol="v3")`: stream rows keep the v2 test; window rows need `source ∈ V3_WINDOWS` and a match outside `NEVER_V3`.

3. **Stream and window strips cannot share one root.** `load_strips` and `read_strip` read a single root, and `check_source` forbids mixing the two sources in one `strips.parquet`.
   - Fix: cut the verified strips into `ball_strips_verified/`. Write a third directory, `ball_strips_v3/`, holding only a combined `labels.parquet`, `labels_codes.npy` (about 26k×6400 int8, roughly 165 MB) and `labels_meta.json`.
   - In that combined table, `file` is `../ball_strips/<f>` or `../ball_strips_verified/<f>`. `root / r.file` resolves these, so `read_strip` needs no change. Keep v2's 300 holdout flags.

4. **The provenance would be wrong.**
   - `provenance_info` hard-codes `exclude=V2_EXCLUDE`, but v3 trains inside that range. It also records the `LR_V2` and `FREEZE_TO` constants, not the values actually used, and has no lineage.
   - Fix: pass `exclude` as a parameter (default `V2_EXCLUDE` so the self-check at L1283 still passes; v3 uses `{}`). Record the real lr and freeze, `protocol: "v3"`, `init {path, sha256, parent train}`, the holdout windows and the whole `labels_meta` (verified constants, sha256 of the input parquets). Add the new script to `recipe()`.

**Major**

5. **The labelling spec has gaps.**
   - Under "-1 everywhere else", 998 tiles would be coded negative: 616 unjudged `judge` tiles, about 347 `not` tiles with p>0.1, and 35 S/B* tiles. Code a 0 disk on each.
   - Put -3 on the N tile's whole ±12 px box (6-7 cells a side), not 3x3 cells (±6 px): the judges cleared the whole 24 px box.
   - Add a 0 ring from ±12 to ±20 px around the bag. The true ball can sit at the box edge, so its blob would otherwise spill onto -1 cells.

6. **Name the input file.** `auto_labels_AB.parquet` has 2,525 "B" tiles and gives 1,940 located frames. `auto_labels.parquet` has 1,058. The "B tiles >24 px apart" drop rule removes 0 frames. Record the sha256 of whichever file is used.

7. **"5 v2 strips in MIX_V2 proportions" cannot be done.** `Sampler` needs whole numbers per pool. Use `MIX_V3 = (verified 8, verified_neg 3, flight 2, action 1, free/cluster 1, free/landmark 1)`.
   - Set the loss and monitor flight masks (L740, L585) to `np.isin(kinds, ("flight", "verified"))`.

8. **Negative strips in located frames.** Making a second positive of the same frame when the ball falls in the crop skews the mix. Instead, place the crop so the ball's ±20 px box is outside it, and skip the frame after 10 tries.
   - Expected yield: 1,022 of the 1,321 located frames have a confident-negative tile more than 200 px from the ball, and 1,085 unlocated frames have one. That gives about 2,100 negatives and 1,940 positives (1,774 train, 166 holdout), so about 4,050 strips.

9. **Resume and deduplication.**
   - `resume_cut` matches finished strips by name, and names contain x0/y0. Draw placements with `default_rng([seed, match, half, kind_id, abs_frame, j])` so a restart produces the same names; otherwise it re-cuts everything.
   - Deduplicate targets by `strip_name` before `cut`. Duplicates produce duplicate rows and codes.
   - Call `merge(..., kinds=("verified","verified_neg"), source=w)`. With the default `KINDS`, or with `("verified",)` as in the code map's snippet, a rerun duplicates rows.

10. **Evaluation.**
    - `--save` writes only at the end of a run, so run one job per M6 window, each saving its own file.
    - U_a/U_b cover all 762 A frames for v2.4000, with 0 empty frames, so reuse works.
    - Also run `ball_v2.4000.pt` in the v3 job (about 13 min more) and check its top-1 matches U_a/U_b. That catches any change in the eval pipeline since then.
    - Never pass `--allow-seen`. `check_unseen` passes on M6 because v3's provenance leaves out 118578.

**Minor**
- **x0/y0 limits:** they must stay within `box[0]..box[2]-320` and `box[1]..box[3]-320`. `place()` already clips to these; any custom negative placement must clip the same way, because `cut()` does not check and the background slice breaks `np.hstack`.
- **Absolute frames:** strips use absolute frames (B's local frame + `start_frame`); the eval's candidate keys are local frames.
- **Window edges:** 16 B frames fall in a window's last 25 frames, where `n_bg` is 7 and there is no background. This matches inference; keep them.
- **132877 at 1084 px:** nothing in the code assumes 1080. Take W and H from the window json and the box from `info["pitch"]` (y-range 92-749).
- **Atomic checkpoints:** save checkpoints to a temporary file and `os.replace` them, then write the provenance, which is the done-marker.
- **Eval threshold:** v3's top-1 only counts if p_raw ≥ 0.05 (`V2_TAU`). Log v3's empty frames.

**Fine-tuning code changes, by file**
- `src/ball_heatmap.py`: the constants in item 2 and `write_provenance(never=)`.
- `train_ball_heatmap.py`: `load_strips(protocol)`, the two flight masks, `optimiser` / `Run` / `train_v2` (init, lr, freeze), `provenance_info`, `recipe`, the `--v3` CLI mode and atomic saves.
- New `build_verified_strips.py`: cutting, labelling and combining.
- New `score_v3_M6.py`: `ev.load_scores` plus `A/frames.parquet` and `A/tiles.parquet`, for the top-1 hit rate and the confirmed-false rate.

**CPU time (4 CPUs)**

| Stage | Estimate | Basis |
|---|---|---|
| Cut ~4,000 strips | 12-18 min | Measured on one window: OpenCV skips frames at 130 fps and fully decodes them at 48 fps. About 1 min per window; the strip count barely matters (about 10 ms each) |
| Label and combine | under 2 min | |
| Train 3,000 steps | 65-75 min | v2 logged 1.26 s per step; v3 trains the early layers from step 1 and the monitor runs about 470 holdout strips |
| Eval, 3 v3 checkpoints + v2 on 762 frames | 50-85 min | Longest window job about 26 min |

**Job plan that survives restarts**

Run the stages one after another: train and eval must never overlap on 4 CPUs. Each stage is skipped if its marker exists, and each runs in the background with a log file.
- **S0:** the three `--check` self-tests, a 20-step `--bench`, and a smoke train with checkpoints at 10 and 20 into the scratchpad, so the provenance code actually runs.
- **S1:** cutting, one loop per window. Done = the window has rows in `strips.parquet` and no partial file.
- **S2:** labels, written atomically.
- **S3:** training. Done = `ball_v3.3000.pt.provenance.json`. A restart begins from scratch, unless you add a resume file every 250 steps (model, optimiser, rng state taken before the next draw).
- **S4:** one eval job per M6 window. Done = its output file exists.
- **S5:** scoring.

**Keep as is**
- Training on B only, with M6 as the selection set.
- Fine-tuning from v2.4000 with the same sigmas (`load_v2` reads them back from the provenance).
- Cutting window strips with `cut` and `window_frames`: the background is limited to the window exactly as at inference.
- Footprint pooling for verified bags (the new kind also escapes `DROP_V2`).
- Keeping some v2 stream strips in every batch.
- The 12 px top-1 hit rule, which matches the bag definition.
- The `ball_v3.<step>.pt` names and the reuse of U_a/U_b.

The two counting scripts are `/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5/crit_feas.py` and `/tmp/claude-0/-home-user-Veo/cad650b3-8188-596d-8724-bf1756dd5cab/scratchpad/ball/v5/crit_feas2.py`.