# SoccerTrack v2: fixed panoramic footage with ground truth

The footage this project is meant for comes from a fixed camera that sees
the whole pitch -- no cuts, no replays -- which broadcast clips (SoccerNet,
the Stoke and Reading windows) are not. Veo's panorama is that view, but
there is no public Veo panorama, and Veo's support pages say only the
follow-cam view can be downloaded. SoccerTrack v2 is the same kind of
footage, filmed by BePro's fixed panoramic systems:

- ten university matches, 934 minutes, both halves, 4K panoramas (4096
  wide for eight matches, 3840 x 1504 for one);
- **game state** per frame: every person's pitch position (metres, origin
  at the centre spot), shirt number, persistent id, role and side;
- **21,428 ball actions** in twelve classes (pass, high pass, cross,
  drive, shot, goal, header, tackle, block, out, throw-in, free kick),
  each with the side and the player who made it;
- per-match calibration; a split (train six, validation M2 and M10, test
  M7 and M9) from the SoccerTrack Challenge 2025.

CC BY 4.0, pseudonymised (numbers and integer ids, no names). Scott et
al., arXiv 2508.01802; https://huggingface.co/datasets/atomscott/soccertrack-v2.

With it the identities, the number reader and the event detector can be
measured on the kind of footage they are for, without clicking: the
ground truth already says who everyone is and who did what.

## Access

The dataset is gated: log in to Hugging Face, accept its terms on the
dataset page, create a read token and set it as `HF_TOKEN` (or
`HF_SECRET`) in the environment; where an outbound proxy adds the token
itself, no variable is needed (`token()` checks `whoami`).

Access is not automatic: until the dataset's owner grants it, every gated
file answers 403 "you are not in the authorized list" even with a valid
token (this happened on 2026-10-08 and was granted the same day).

## Reading it

`src/soccertrack_v2.py` reads it without downloading whole files -- a
half's game-state file is ~2.7 GB of JSON and its video a 45-minute 4K
panorama:

- the game state is streamed and flattened once per half into a parquet
  table in `.cache/soccertrack_v2/gsr` (`gsr_table`);
- the events are read whole (`events`), and `with_actors` adds the actor's
  shirt and position from the game state;
- the video is read by HTTP range requests, so the decoder seeks straight
  to the minutes wanted (`RemoteFile`, `read_frames`, `cut_clip`): on a
  local clip, frames read after a seek were identical to OpenCV's, and
  2.4 MB of 24 MB was fetched for five frames.

The released files differ from the repository's own format notes (a
SoccerNet game-state file rather than flat records, the frame number
inside a string id, attributes nested, capitals in event labels, event
times counted from the match start before release v1.2); the module
docstring lists what is read. The direction of the pitch `y` axis is given
both ways in the dataset's documents; `y_towards_camera` settles it from
the data when a half is first read.

`fetch_soccertrack_v2.py` cuts a window and its ground truth into
`data/soccertrack_v2/`: the video (H.264, full resolution, so it plays in
the labelling pages), every person on every frame (`_gt.parquet`), the
ball actions with their actors (`_events.csv`) and where it came from
(`.json`).

    python fetch_soccertrack_v2.py --list
    python fetch_soccertrack_v2.py --match 117093 --half 1 --start 10 --minutes 3

`python -m src.soccertrack_v2` checks the reading on synthetic files in the
shipped format (a CI control). A half's GSR streams in ~165 s; a 3-minute
window of video is cut in ~210 s.

## What the real files are like

Checked on M2 (117093) and M3 (118575), first half, minutes 10-13:

- **Frames** are 4096 x 1080 (the `images` sizes are right in v1.2). The
  panorama is stitched and strongly curved: the near touchline bows down
  through the middle of the frame, and the far-side players are 35-45 px
  tall.
- **Pitch y grows towards the camera** (the dataset README is right, its
  format doc wrong): on both windows y correlates with the image row of the
  feet (0.93 on M2). The first check, by box height, said the opposite,
  because **every released box is 42-43 px tall**; it now reads the feet.
- **The boxes are approximate.** They are 17-81 px wide (median ~45 px)
  against ~18 px for a detected player, and their bottom centre sits a
  median ~10 px from the detected feet, with no time offset (agreement is
  best at shift 0 within +-50 frames). Drawn on frames they sit on or
  beside their player; some players stand partly outside their box.
- **Only players and goalkeepers** are in the GSR: no referees, no ball.
  98.7% of frames hold exactly 22.
- **BAS `frame` is 1-based** (as are the GSR image ids); the loader took it
  as 0-based, one frame late. Fixed.
- Both windows hold 22 people; M2 has 86 ball actions (35 passes, 30
  drives, ...), M3 72. Every action's actor was found in the game state.
- The release's **pitch lines** give the pitch outline in pixels; drawn on
  the frame, the far side and the ends sit on the painted lines, the near
  touchline is drawn as straight segments up to ~25 px inside the curved
  painted one. `fetch_soccertrack_v2.py` stores the outline in the
  window's `.json` (`--pitch` adds it to a window already cut).

## The pipeline on a fixed panorama

`run_pipeline(..., fixed_camera=True)` (`score_soccernet.py`) is the
broadcast pipeline with three changes; the broadcast defaults are
untouched:

- players are detected at the frame's own width (`imgsz` 4096): shrunk to
  1280, a far-side player is ~13 px tall and yolov8s finds 55% of the
  players on M3 (18 people a frame) against 95% at full width. This is
  tiling without seams, at the same cost;
- camera motion is neither measured nor compensated (replay detection was
  never in this path);
- with `pitch=` (the outline) and `pitch_margin_px`, player detections
  whose feet are outside the outline by more than the margin are dropped
  before teams are assigned (`player_filter.inside_pitch`).

`eval_soccertrack_v2.py` runs it on a window and scores it.

**Settings were chosen on M3 and reported on M2.** On 30 frames of M3,
yolov8n / s / m at full width found 93 / 95 / 97% of the players at
confidence 0.25, precision 0.75 / 0.80 / 0.77 (0.6 / 1.5 / 4.2 s per
frame on 4 CPU cores); yolov8s, every second frame, was taken. Confidence is still chosen by the
pipeline's own probe (0.25 on M2, 0.30 on M3). The pitch margin was chosen
on M3's detections by a rule fixed beforehand -- the best precision whose
recall is within 0.005 of no cut:

| margin (M3) | recall | precision | people / frame |
|---|---|---|---|
| no cut | 0.940 | 0.843 | 24.6 |
| 0 px | 0.929 | 0.934 | 21.9 |
| **10 px** | **0.940** | **0.916** | 22.6 |
| 25 px | 0.940 | 0.900 | 23.0 |
| 50 px | 0.940 | 0.873 | 23.7 |

## Results

Every second frame of each window (2,250 frames, 49,500 ground-truth
boxes). A detection matches a ground-truth person when its feet are within
43 px (one box height) of the box's bottom centre, one to one per frame.

**IoU >= 0.5 does not work against these boxes:** it matches 4-5% of
them on both windows whatever the detector, because a box two and a half
times wider than the player caps the IoU of a perfect detection near 0.4.
Those numbers are written by the scorer and not repeated here.

Precision counts the referee and the two assistants against us (the
truth has no officials): up to 3 of the ~23-28 detections a frame, so it
is understated by up to ~0.1 (not measured separately).

**M2 (117093, the reported window):**

| stage | recall | precision | identities | people split | purity | IDF1 | teams |
|---|---|---|---|---|---|---|---|
| detections, no outline | 0.941 | 0.532 | 264 | 19.3 | 0.75 | 0.30 | |
| after re-joining and roster, no outline | 0.536 | 0.496 | 98 | 8.1 | 0.71 | 0.33 | 0.50 |
| detections, outline + 10 px | 0.935 | 0.743 | 190 | 19.1 | 0.75 | 0.36 | |
| after re-joining and roster, outline | 0.589 | 0.673 | 70 | 8.3 | 0.69 | 0.38 | 0.51 |

**M3 (118575, where the settings were chosen):**

| stage | recall | precision | identities | people split | purity | IDF1 | teams |
|---|---|---|---|---|---|---|---|
| detections, no outline | 0.940 | 0.843 | 174 | 15.9 | 0.81 | 0.42 | |
| after re-joining and roster, no outline | 0.899 | 0.870 | 89 | 12.4 | 0.78 | 0.44 | 0.77 |
| detections, outline + 10 px | 0.940 | 0.916 | 152 | 15.9 | 0.81 | 0.44 | |
| after re-joining and roster, outline | 0.916 | 0.932 | 87 | 12.8 | 0.78 | 0.45 | 0.77 |

("people split": over how many of our identities one real person is
spread, frame-weighted, 1 is perfect; "teams": share of matched detections
whose team, mapped one-to-one to the sides, is the person's side.)

The "teams" column above counted goalkeepers and let a third label stand
in for a team. The scorer now reports, over outfield players only, team
accuracy (among detections given team_A or team_B), coverage (the share
given either) and team recall (outfield boxes detected, kept and on the
right team). With the outline, M2: accuracy 0.51, coverage 0.95, team
recall 0.27; M3: 0.79, 0.95, 0.69.

What this says:

- **Detection works**: full-width yolov8s finds 94% of the players on both
  matches.
- **Team assignment failed on M2 (chance) and was weak on M3 (0.77)**
  -- fixed for the bench case, see "Teams on a fixed view" below. On
  M2 the two kits have the same hue -- one saturated blue, one white
  (shirt saturation 0.93 against 0.39) -- and the substitutes and staff
  standing at the far touchline, inside the 10 px margin, are as many
  tracks (45) as a team (37 and 40). k-means put both teams in one cluster
  and the bench in the other, so `team_assignment_v2`'s rule "the two
  largest clusters are the teams" names the bench a team. The roster cap
  (11 per team per frame) then drops about half the real players, which
  is the fall from 0.94 to 0.59 recall. On M3 one kit is dark (value 0.35)
  and 14 of its 62 tracks are put in "other".
- **Identities fragment badly.** Each real person is spread over 13-19 of
  our identities (frame-weighted) and IDF1 is 0.36-0.45. Re-joining roughly
  halves the number of identities without making them purer.
- **`identify_players.py` was not run**: it needs `.cache/player_reid.pt`
  (with `--split`) or `.cache/jersey_reader.pt`, and `.cache` held neither
  in this container. The identities above are the tracker's.
- **No pitch coordinates**: `ground_plane` finds no plane (its focal length
  disagrees with itself across baselines, spread 0.68-0.80), as expected of
  a stitched, curved panorama, so positions stay in pixels and goals,
  shots and out-of-play stay disabled. The events the pipeline emits (11
  on M2, 0 on M3) were not scored.
- **Speed**: ~1.2 video frames a second (every second one detected) on 4
  CPU cores; a 45-minute half at this setting would take ~16 hours here.

## Teams on a fixed view

The failure above has a cause particular to a fixed whole-pitch view: the
bench, the staff and the assistant referees are filmed all match, at the
pitch's edge, and on M2 they were as many tracks as a team, so "the two
largest colour clusters are the teams" named them a team. With a pitch
outline, `run_pipeline(fixed_camera=True, pitch=...)` now fits the kit
clusters only on tracks whose feet stay at least 0.15 player heights
inside it (median over the track, `player_filter.track_depth`); every
other track is "other", however many there are
(`team_assignment_v2.cluster_teams`, `fit_tracks`). It also warns when a
"team" has more than eleven players on at once -- both teams in one
colour cluster. Broadcast is unchanged (identical team column on M3 with
the default path); the kit features and depth of every track are written
to `team_features.json` so rules can be compared without the video.

**How it was chosen.** Three independent proposals and a critic were
weighed first (a design panel); the rules were then fixed before any
result: the feature by pooled outfield team recall over four tuning
windows (M3 and new windows from the training matches M4 118576, M6
118578, M8 128058, two minutes each, first half from minute 20), the
simplest within 0.005 winning; the core depth D as the largest on
{0, 0.05, 0.1, 0.15, 0.2, 0.3} within 0.005 of the best; stop if the core
cost more than 0.03 on any window; and the bench stress test -- every edge
track copied four times -- must leave the team partition unchanged. M2
was already diagnosed, so it is reported as that case; M10 (132877, the
other validation match, minutes 10-13) is the held-out result.

Team recall (outfield boxes detected, kept and on the right team) on the
tuning windows:

| kit feature, core | M3 | M4 | M6 | M8 | pooled | stress test |
|---|---|---|---|---|---|---|
| hsv4, none (before) | 0.685 | 0.769 | 0.745 | 0.585 | 0.695 | fails |
| **hsv4, D 0.15** | 0.699 | 0.769 | 0.737 | 0.560 | 0.692 | passes |
| chroma, none | 0.736 | 0.769 | 0.279 | 0.581 | 0.607 | fails |
| chroma, D 0.15 | 0.726 | 0.769 | 0.731 | 0.541 | 0.696 | passes |

The candidate kit feature, "chroma" (s cos h, s sin h, v: hue counting
only as much as the kit is saturated), was proposed because M3's dull kit
fragments on hue noise, and it lifts M3 by 0.05; but on M6 it merges both
teams into one cluster unless the core is used, it is worse on M8, its
core costs 0.04 there (over the stop rule), and pooled it is within 0.005
of the existing feature. It was not adopted (`kit_feature="chroma"`
remains an option). The core's largest cost is 0.025 (M8: tightening the
kit clusters lets the unchanged 2.5x residual cut refuse a few more
players; only 2.7% of M8's outfield rows are in edge tracks).

Report windows, the frozen rule against the one before it (same
detections; "non-outfield in a team": detections kept in a team that are
not an outfield player, per frame):

| window | rule | team recall | team accuracy | coverage | non-outfield in a team | detection recall |
|---|---|---|---|---|---|---|
| M10 (held out) | before | 0.804 | 0.891 | 0.978 | 1.26 | 0.921 |
| M10 (held out) | **frozen** | **0.800** | 0.893 | 0.958 | 0.26 | 0.933 |
| M2 (diagnosed) | before | 0.266 | 0.509 | 0.952 | 6.25 | 0.589 |
| M2 (diagnosed) | **frozen** | **0.665** | 0.779 | 0.974 | 1.38 | 0.886 |

(Chroma with the core, for the record: M10 0.800, M2 0.462.)

So: the bench can no longer be named a team, which was M2's failure, and
the people kept in teams who are not players fall four- to five-fold; on
windows without that failure team recall is unchanged (M10 -0.004, the
tuning windows -0.003 pooled). It does not make team assignment accurate
in general: about one matched outfield row in ten is still on the wrong
team on M10 and more than one in five on M2, where the merge warning still fires
(a "team" with a median of 12 on at once: about 30% of one team's rows
join the other kit).

## Next

1. **Re-identification from the ground truth**: SoccerTrack v2 gives
   persistent ids on six training matches -- free training pairs for
   `train_player_reid.py`, on this very kind of footage, so the `--split`
   identities can be measured here.
2. **Pitch coordinates**: map the feet through the release's calibration
   (`raw/<match>/` homography and distortion maps), or a one-off clicked
   calibration for a Veo panorama, instead of `ground_plane`'s pinhole
   model; then score events against the BAS actions.
3. A score that tolerates the boxes: the foot distance above, or the
   pitch positions once (2) exists.
4. **Teams, what is left**: giving every track its true majority side --
   the best any per-track rule can do -- reaches team accuracy 0.84 on
   M3, 0.88 on M10 and 0.82 on M2, because tracks switch from one player
   to another (see identities). The frozen rule is at that ceiling on M10
   (0.89; it refuses some impure tracks) and 0.04 below it on M2, where
   two kits of one hue are still close (the merge warning fires). Better
   teams now need tracks that stay on one player, not a better colour
   rule.
