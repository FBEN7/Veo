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
- Per-player stats need identities. The owner's choice: the coach names
  players on clips cut by the product now; automatic re-identification later.

**Status doc (French, update it after each result):** Claude Docs at
https://claude.ai/code/artifact/0abccf0d-8aa5-4159-b47f-7431c2771eaa.
- Body node `46cfe67c-2f48`, block prefix `mwy08bzh1q7`, last seen rev 62.
- Tables:
  - module table `.1860`;
  - stats table `.11947` (status enum `c07b100f-0db3`: 0 Abouti, 1 Partiel,
    2 Non abouti; priority enum `4e6d9647-5507`);
  - blockers `.6487`;
  - dated log `.18125` (newest first, insert after header row `.18126`).

## Results so far (all in SOCCERTRACK_V2.md)

| Topic | Result |
|---|---|
| Learned pass/carry spotter (`research/session_scratchpad/ball/events_learn/`) | M1 test F1 0.660 vs rules 0.441, +0.219 [+0.166, +0.271]: shown. Passes: recall 0.29 → 0.77. A second use of M1, so weaker evidence. |
| Calibration | Release calibrations are right (truth-box feet 0.3-0.5 m). Our detections' foot point is off (0.9-1.4 m). M1's labels sit in another pixel frame. The M1 refit (`calib_fix/`) is a non-physical warp, for player feet only. |
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
4. Then: a foot-point
   correction learned against truth boxes, and the identity work.
