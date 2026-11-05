# Handoff (2026-10-20): read this first

Written by the previous Claude session for the next one. The owner is François.

## Working with the owner

- Reply in **French**, short, critical (no answers to please). Ask clarifying
  questions before long answers.
- Cost-sensitive: avoid multi-agent workflows (they hit the session limit).
  Keep CPU/GPU spend proportionate.
- Never print or commit a token. Never disable TLS or unset HTTPS_PROXY. No
  model identifier in commits or PR text.
- Commit trailer:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: <your session URL>`.
- Branch `claude/game-analysis-veo-8kze2z`. A stop hook wants a clean, pushed
  tree at the end of every turn. `.cache/` and many outputs are gitignored:
  add them with `git add -f`; never commit `__pycache__`.
- Long jobs run as tracked background tasks (2-h cap; relaunch, everything
  resumes). The VM suspends when idle, and that kills untracked processes.
- The 4 cores are shared: never run two heavy jobs at once.

## The project

The "match-report" product gives amateur clubs automatic match statistics
from one fixed panoramic camera. Research uses SoccerTrack v2 (10 university
matches, M1-M10; loader `src/soccertrack_v2.py`; notes `SOCCERTRACK_V2.md`,
the long, detailed record).

The 15 target statistics are listed in the status doc.
- Priorities: passes and carries, then shots and goals.
- Per-player stats need identities. The owner first chose: the coach names
  players on clips cut by the product; on 2026-11-01 asked for zero effort
  instead. Identity by role fails its ceiling (0.59 of actions, `identity_roles/`).
  **2026-11-02: team statistics only for now** (owner); per-player stats wait.

**Status doc (French, update it after each result):** Claude Docs at
https://claude.ai/code/artifact/0abccf0d-8aa5-4159-b47f-7431c2771eaa.
- Body node `46cfe67c-2f48`, block prefix `mwy08bzh1q7`, last seen rev 78.
- Tables:
  - module table `.1860`;
  - stats table `.11947` (status enum `c07b100f-0db3`: 0 Abouti, 1 Partiel,
    2 Non abouti; priority enum `4e6d9647-5507`);
  - blockers `.6487`;
  - dated log `.18125` (newest first, insert after header row `.18126`).

**Bilan d'une page (2026-11-02, French, for the owner's decision):** Claude Docs at
https://claude.ai/artifact/QSjBMF2krjrkoNLAWu86U6 (nothing shown as a match
statistic; options: semi-automatic product, players near the ball, new data).

**Plan (2026-11-05, French):** Claude Docs at https://claude.ai/artifact/W3p48YE4osJht9DBKRbjYf. The owner wants a 100% automatic report, has no pilot club or customer input yet, and has time. Plan: (1) target-camera footage from 1-2 pilot clubs (GDPR consent) and the clubs' top statistics, while I measure the chain with no annotation on SoccerTrack (automatic pitch outline); (2) owner annotates 3-5 target matches with a tool I build, 2 kept unseen; (3) adapt and test against product bars fixed beforehand. **2026-11-05 updates:** the owner will use a panoramic camera of SoccerTrack's type, no pilot club for now; product bars validated (goals all found, <= 1 false per match; shots 80% found, 80% right; possession within 5 points per half; passes within 10%; distance within 5%); a one-off setup per pitch at installation (about ten clicks: outline and calibration) is acceptable.

## Results so far (all in SOCCERTRACK_V2.md)

| Topic | Result |
|---|---|
| Learned pass/carry spotter (`research/session_scratchpad/ball/events_learn/`) | M1 test F1 0.660 vs rules 0.441, +0.219 [+0.166, +0.271]: shown. Passes: recall 0.29 → 0.77. A second use of M1, so weaker evidence. |
| Calibration | Release calibrations agree with the truth (truth-box feet 0.3-0.5 m). M1's labels sit in another pixel frame. The M1 refit (`calib_fix/`) is a non-physical warp, for player feet only. |
| Foot point (`footpoint/`, `footpoint/diag/`) | A learned correction fails (match-specific offsets). Diagnostic: no timing lag; our boxes sit on the feet in pictures, the truth's boxes are approximate (fixed height, 10-25 px off). The 1.2-1.7 m gap is mostly the truth's; **this truth cannot measure our position error below ~1.2 m**. Real error needs owner foot clicks. |
| Distance | A 1 s estimator holds (2-4%, partly circular). |
| Possession | Not better than a constant 0.5. |
| xG (models/xg_xghub.json) on GT shooter feet | AUC 0.70 [0.60, 0.79]. |
| Goals by kick-off formation (`research/session_scratchpad/shots_goals/`) | GT dev 26/26, 0 false runs. M7 2nd detections 5/5. Frozen in `frozen_goals.json` (T 0.85, N 2, kick_spot). **Registered test done: fails** (`goals_test.json`): 4/8 goals, 2 false runs, team right 2/4 (the other rule 4/4). M1 3/3; M9 0/3: kit clustering did not separate the teams. SOCCERTRACK_V2.md and the status doc are updated (rev 52). |
| Shot ball gate | GO: 19/26 reachable (M7 3/10, M9 16/16; placebo 0.07/0.15). `ball_gate.json`, `gate_recheck.json`. |
| GPU on Kaggle (`research/session_scratchpad/kaggle/`) | Works: T4 x2, ~14.5 min per 3-min window (CPU ~2.5 h), ~10 min per short clip. Identical to CPU on 7 windows (players, teams, events_v1, v3 rows). `push_run.py` + `run_windows.py`; `cut_shot_clips.py`; `opened.json` for sealed clips. ~8 h of the 30 h weekly quota used on 2026-10-20/21. |
| Learned shot spotter (`shots_learn/`) | Registered test on M5: **fails by one shot**, recall 17/33 (needs 18), precision 0.61; dev CV 0.54 / 0.52; the rule gets 2/33. `test.json`. |
| Goals revisited (`goals_v2/`, `teams/`) | Teams are right (scan 0.977 on M9); the registered rule is right on the ground truth (41/41, 0 false); the misses come from the detections' share. A centre-spot variant failed its M1 check (0/3 vs 3/3). Per-detection smoothed teams beat per-track (0.92-0.98 vs 0.79-0.93). |
| Who made the pass (`passer/`) | Nearest player to the v3s_link ball at the label: inside the clips' ball range (frames 350-1075) right on 49% of actions (passes 50%), 72% when it names someone (the first 25% counted the margins, which have no ball: `ball_range.json`). **Default rule now: possession (`src/passer.py`, rule C, -0.4/+0.2 s): 56.8% (passes 58.6%), in-sample; registered M7/M9 test fails: 34.8% (M7 53%, M9 25%). After turning off the grass tests on fixed views (players and ball, both dropped M9's real objects): M7/M9 58.2% (M9 60.7%), exploratory.** Ball gap filling gives nothing; wrong ones are mostly contests with opponents and undetected passers (`diagnose.json`). M1 not measurable (truth matching ceiling 28%); M7/M9 lack a linked ball. |
| Learned shots v4 (`shots_learn/v4/`) | v2 + players' reactions (keeper proxy, box players, collective stop), no grass tests: 165 hits / 113 false / 104 missed (recall 0.61, precision 0.59) vs 150 / 126 / 119; +15 hits, 95% [-3, +34]. Cross-validated only. Missed shots lack the ball in flight toward goal (`shot_flight/`): a detector issue (GPU). |
| Ball detector v4 (`research/session_scratchpad/ball_v4/`) | v3b fine-tuned on 554 owner-clicked shot frames (match folds A/B, Kaggle T4 x2, 50 min). Shot balls after the kick: hit@4 0.338 → 0.451, +0.114 [+0.080, +0.150] (317 frames). **Registered rule fails on the M6 guard**: hit@4 1.000 → 0.955 (v4A) / 0.970 (v4B), hit@1 0.955 → 0.865 (other objects ranked first); more confident on empty crops. Not adopted (`results/decision.json`). **v4r** (half learning rates, more stream strips, distillation to v3b; fresh M6 guard, `protocol_v4r.json`): shots +0.095 [+0.064, +0.127], guard hit@4 within 1.1 points but hit@1 0.941 → 0.872 and empty crops 0.863 → 0.918: **fails too**, not adopted (`results_v4r/decision.json`). |
| Shots hybrid ball (`shots_learn/hybrid/`) | v3b everywhere, v4r within 2 s of likely shots (62% of the ball range), spotter v4 re-cross-validated: 157 found / 138 false vs 165 / 113, hits -8 [-26, +8]: **fails**, not adopted (`decision.json`). The ball after the kick is better tracked (missed shots heading to goal 0.45 → 0.53) but the spotter does not gain. Control (same steps, v3b rows): 167 / 124, so plumbing alone moves a match by up to 5 hits. |
| Shots + candidate cues (`shots_learn/cand/`) | v3b track kept, 7 cues from raw candidates near the goals: from v4r (C) vs from v3b (B): hits -1 [-11, +9], **v4r adds nothing**. B vs v4 spotter +10 [0, +22] hits but +14 false alarms: not shown. Retraining the ball detector for shots is closed (`decision.json`). |
| Player on the ball (`near_ball/`, option B, owner) | Image-space diagnostic, 2,257 actions, 9 matches: actor detected 0.767 (others on the same frames 0.884), taken by a neighbour 0.113, missed 0.121; merged boxes 0.097; detected actors 1.59 m median from the truth (28% beyond 3 m). **Detector variants** (`plan_v2.json`, GPU ~1.2 h of 10 h): raw yolov8s already finds the actor on 0.867; best within the false-detection guard yolov8x +0.020 [+0.009, +0.030], under the 0.03 bar: **no new detector**. Exploratory: 11.3% of actions are found by the raw detector and lost by the pipeline (`reading_v2.json`). **Stages** (`plan_v3.json`, CPU): the tracker stage loses 10.2 of those 11.4 points (roster cap 0.9); on 7 of 9 matches that stage also held a since-removed grass test on players (plan_v4 addendum 1 separates them): ByteTrack writes only detections matched to an activated track, and drops the actor's box in contact. Lever: write every detection (`reading_v3.json`). **Tracker variants** (`plan_v4.json` + addenda, GPU 2.6 h + Kaggle CPU 3.4 h, `v4.json`): every detection kept (B2) vs the pipeline replayed (A), actor detected at S4 0.878 vs 0.847, +0.032 [+0.026, +0.038], 9 of 9 matches; guards hold; bar met by 0.2 points. Teams for short tracks alone +0.003. But A does not reproduce the original on M7/M9 (0.804 vs 0.778 at S4, M7 +12.8 at S1, M9 -1.6; `reading_v4.json`), so the grass test's cost on the 7 other matches (+0.088) is confounded; and B2's added boxes are one-frame ids (same track ±1 s 0.756 vs 0.814). Not in the pipeline yet: join those boxes to tracks, explain the replay gap. **Replay gap explained** (`plan_v5.json`, `v5.json`, local CPU): the original runs feed the tracker boxes above each clip's auto-tuned conf_player (0.08 to 0.5, median 0.4), the replay 0.20; at each clip's own threshold M7 is reproduced exactly (0.663 = 0.663, boxes 98.5%+ shared). So plan_v3's tracker-stage loss is mostly the threshold (ByteTrack at 0.20 costs 0.8 points), and the lever is conf_player (M7 +12.8 at S1); its false-detection cost on the 9 matches is not measured yet; M9 (imgsz 3840) not covered. **Fixed 0.20 vs auto-tuned threshold at S1** (`plan_v6.json`, `v6.json`, local CPU, 2,230 actions): actor detected 0.859 vs 0.779, +0.081 [+0.063, +0.102], 9 of 9 matches, bar met; but the false-detection guard fails at S1 (5.93 vs 3.80 unmatched boxes per action frame, 1.56x); next per the plan: the same at S4 (Kaggle CPU ~3.4 h). Exploratory (`reading_v6.json`): the per-clip replay matches the original on 98.9% of actions; the old grass test on players cost 0.3 points. **Same at S4** (`plan_v7.json`, `v7.json`, Kaggle CPU 52 min): 0.847 vs 0.770, +0.077 [+0.060, +0.096], 9 of 9 matches, but the false-detection guard fails again (5.58 vs 3.61 unmatched boxes per action frame, 1.54x): **bar not met, no pipeline change**. Open: what the extra boxes are and what they cost in the report's figures; any new threshold needs matches it was not chosen on. Exploratory (`reading_extra_boxes.json`, `reading_actor_team.json`): the extra boxes (2.0 per action frame) are 90% outside the outline and 96% labelled 'other' (touchline people, not in the truth); team-labelled unmatched boxes 0.74 vs 0.59 (1.24x); actor detected with a team label 0.737 vs 0.675. A guard on what the report uses would need its own plan. **Closed (owner, 2026-11-05): the auto-tuned threshold stays.** No other annotated matches; M1 cannot be mapped into our video's pixels (`m1/reading_m1.json`: best held-out control 0.52 vs 0.89, nothing matched near the camera). Local caches of the GPU detections and tracker variants were deleted; they remain as the outputs of the Kaggle kernels veo-nearball-detall, veo-nearball-track and veo-nearball-track-v7. |
| Distance per team (`team_distance/`) | 1-s speeds of our tracks x 10 outfield players, per-detection kit teams, 249 clips: 9.6% off against 5.0% for a constant rate per team-second: **worse, fails**. Ours is below the truth in all 20 team-matches (1-24%); the truth's teams do not help (12.1%). The share between teams is ~0.5 everywhere. |
| Passes per team (`team_passes/`) | Learned spotter on the 249 shots clips' ball range, nested by match: a team's share of the passes is no better than 0.5 (error 0.087 vs 0.097), its pass success rate is worse than a constant (0.199 vs 0.061): **fails**. Our passes carry the right team 68% of the time. **Event-team rules** (nearest at the frame, possession actor's track, team vote): 0.70 / 0.68 / 0.69, none better (`plan_v2.json`). The limit is the players' team labels near the ball: the truth actor's nearest detection has a team on 52%, right on 79% of those. **Kit teams per detection** (Kaggle CPU kernel, `plan_v3.json`): detections right among labelled 0.85 -> 0.895, events 0.70 -> 0.735; share error 0.071 vs 0.097 [-0.086, +0.029], success worse than a constant: **still not shown**. 38% of actions have no detection within 3 m of the actor. |
| Chain ball -> player -> team (`chain/`, 2026-11-05) | 2,257 actions, 223 clips, 9 matches (M7/M9 now development data): the team of our player nearest our ball is the actor's on 0.345 [0.324, 0.368]; our ball within 3 m on 0.506. Swapping in the true ball +0.252, the true team +0.008: **the ball is the link to work on first**. Exploratory: a ball row exists but lies > 3 m away on 31% (median 5.9 m), none at all on 12%. **Selection or detection** (`plan_v2.json`, `cand.json`): linked ball within 3 m 0.506; a frame's top candidate within 3 m but not linked 0.150; a lower-ranked candidate 0.087; no candidate 0.257; selection minus detection -0.020 [-0.069, +0.032]: both matter; the linking step is the cheaper (CPU). Exploratory (`reading_sel_split.json`): of the 23.7% selection, 14.1 points have the near candidate below the linker's tau 0.10 (median 0.072), 9.7 points at 0.10 or more. |
| Identity by role (`identity_roles/`) | Zero-effort identity, ceiling from the truth (8 matches, full halves): a player's role in the team's shape names the right actor on 0.587 of actions (0.756 if followed 60 s); bar 0.70: **not viable**, stopped before our detections. |
| Shots spotter measure (`shots_learn/eval/`) | Event AP of leave-one-match-out peaks, probabilities averaged over 3 seeds (`eval_ap.py`). Null change (control - v4): AP +0.006 [-0.020, +0.038], half the counts' interval (recall [-0.052, +0.063]). **Now the measure for spotter comparisons** (`decision.json`). Track noise remains: a null change moves M9 by 0.17 AP; a real gain needs about +0.04 AP. Re-scored, exploratory: hybrid -0.004 [-0.046, +0.046], cues B +0.026 [-0.013, +0.054], C +0.020 [-0.020, +0.057]. v4 AP 0.612. |
| Shots (`protocol_shots.json`) | **Dev fails**: frozen 40 km/h / 35 m / mouth 3 m, F1 0.053 (1/32). Diagnostic: the rule is too strict for the v3 ball, the team filter halves what is left. **M1 test not run** (owner decision B, `shots_decision.json`). |

## Registered protocols: follow them exactly

All are hashed (`*.sha256`).
- `shots_goals/protocol.json`, with `protocol_notes.json` and
  `gate_recheck.json`.
- `shots_goals/protocol_shots.json`: closed (dev failed, test not run;
  `shots_decision.json`).
- `kaggle/backend_note.json` (+ `backend_note_addendum1.json`): the GPU
  backend, hashed.
- M7 (128057) and M9 (132831) are sealed test matches. Opening them needs the
  `VEO_UNSEAL` key (given in your session prompt, not in the repo); every
  opening is logged in `src/unseal.py`.

## Next steps

1. **The GPU is ready** for heavy runs: `push_run.py W1 W2:seen ... --data
   <dataset>` (W:seen = ball with --allow-seen). Upload videos as a private
   dataset first (see how `veo-st2-dev1` was built). Two kernels can run at
   once. `kaggle kernels logs` is empty while a run is live; the owner can
   watch the Kaggle page.
2. Shots: the learned spotter (shots_learn/) misses the M5 bar by one shot;
   more training shots or a better ball near the goals are the levers. Teams
   per track would help both goals and shots. Clips are cut here
   (shots_learn/batches.py) and run on Kaggle; Kaggle secrets do not reach
   API-pushed kernels.
3. Who passed (`passer/`): default `src/passer.py` (possession). What
   is left: contests and passers the detector misses (GPU), or a model
   learned on verified ball positions. M7/M9 need their
   metric step (video) on Kaggle (CPU kernel, push_metric.py) to be scored.
4. Then: the identity work. (A foot-point correction is done and fails;
   the gap is the truth's, see `footpoint/diag/reading.json`.)
