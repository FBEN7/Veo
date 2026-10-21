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
- Body node `46cfe67c-2f48`, block prefix `mwy08bzh1q7`, last seen rev 47.
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
2. Shots need a new detector and a new registration (see SOCCERTRACK_V2.md,
   "Shots"); teams per track would help both goals and shots.
3. Then: actor attribution of learned events (who passed), a foot-point
   correction learned against truth boxes, and the identity work.
