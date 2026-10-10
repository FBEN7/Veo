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

## Re-identification from the ground-truth ids

`identify_players.py --split --reid W` cuts tracks where the look changes
and joins pieces by look (`track_split`, `player_identity`). Its embedding
had been trained on broadcast and its weights were not here, so the plan
was to train one on SoccerTrack v2 itself, whose persistent ids give
labelled people for free.

**The data, and a leak that had to be closed.** `build_reid_crops.py`
cuts each person's crops as the pipeline cuts them (our detections,
matched to a ground-truth person by the feet; a label only when the match
is within min(43 px, 0.6 x height) and no one else is within twice that:
29-56% of matches kept). The dataset's `player_id` is the same person in
every match, and the same squads recur: M4 shares 10 people with M3, M1 7
with M2, M8 5 with M10. So everyone seen in the first halves of M2, M3 or
M10 (67 people) is dropped from training altogether, and the loader
asserts it. What is left: 60 distinct people from five training matches
(M1, M4, M5, M6, M8; both halves of three of them, pooled per match by
`player_id`), about 3,000 crops, mostly 31-95 px tall (median 49). M3 is held out for the
thresholds, M2 and M10 for the report.

**Training did not work.** The broadcast recipe (batch-hard triplet,
0.3 hinge) collapsed: after 3,000 steps the median cosine between crops
of *different* people was 0.95 (0.62 untrained), the loss sat at the
margin, and joining no longer depended on its threshold. Picking a fix on
M3 would have used M3 twice, so recipes were compared on an internal
validation match instead (M5, its people also kept out of training), for
1,000 steps each:

| recipe | M5 mAP among team-mates | median cosine, different people |
|---|---|---|
| untrained (ImageNet body, seed-0 head) | 0.183 | 0.64 |
| batch-hard, 0.3 hinge, lr 3e-4 | 0.183 | 1.00 (collapsed) |
| batch-hard, soft margin, lr 1e-4 | 0.185 | 0.97 |
| batch-all, soft margin, lr 3e-4 | 0.165 | 0.84 |
| batch-all, soft margin, lr 1e-4 | 0.167 | 0.82 |

On the internal validation match no recipe beats the untrained features
by more than 0.002 mAP, all lose on rank-1, and the ones that do not
collapse are worse. The collapsed 60-person model does rank crops better
on M3, M2 and M10 (below), but its similarities are all near 1, so no
threshold separates people with it, and end to end it is worse. So the
embedding used is the untrained one (`train_player_reid.py
--save-untrained`, reproducible). Crop-level, rank-1 among same-side
outfield team-mates (about ten candidates, so chance is ~0.1), with
near-in-time copies left out:

| window | untrained | untrained body, 512-d | trained (collapsed) |
|---|---|---|---|
| M3 (thresholds) | 0.29 | 0.35 | 0.39 |
| M10 (report) | 0.27 | 0.29 | 0.34 |
| M2 (report) | 0.20 | 0.18 | 0.23 |

(The collapsed model ranks crops better -- on M3 mAP 0.29 against 0.19
-- but with every similarity near 1 a threshold cannot use that: end to
end it is worse, below.)

**End to end.** The cut and join thresholds were chosen on M3 for each
embedding by a rule fixed in `tune_st2_split.py` (the highest IDF1; within
0.005, no cut first, then the lowest cut, then the highest join): cut 0.7
and join 0.85 for the untrained embedding. The real `identify_players`
path reproduces the tuner's M3 IDF1 (0.521).

| window | identities from | IDF1 | identities (22 real) | people split | purity |
|---|---|---|---|---|---|
| M10 (held out) | the tracker (re-joined) | 0.468 | 86 | 16.9 | 0.80 |
| M10 | **split + join, untrained, 0.7 / 0.85** | **0.524** | 50 | 14.3 | 0.70 |
| M10 | split + join, untrained, 0.4 / 0.7 (broadcast) | 0.520 | 38 | 14.1 | 0.69 |
| M10 | split + join, trained (collapsed), 0.9 / 0.97 | 0.511 | 46 | 16.2 | 0.71 |
| M2 | the tracker (re-joined) | 0.375 | 91 | 14.6 | 0.67 |
| M2 | **split + join, untrained, 0.7 / 0.85** | **0.408** | 58 | 13.7 | 0.61 |
| M2 | split + join, untrained, 0.4 / 0.7 (broadcast) | 0.392 | 43 | 12.6 | 0.58 |
| M2 | split + join, trained (collapsed), 0.9 / 0.97 | 0.391 | 42 | 13.2 | 0.58 |

So `identify_players.py --split` with the untrained embedding raises IDF1
by 0.03-0.06 on the report windows, but by joining: identities halve
while purity falls by 0.06-0.10 -- some joins are of two people. The
people are still spread over about fourteen identities each, and the
cuts barely fire (on M3, 91 tracks become 118 pieces at 0.7). Re-
identification by appearance is not what limits identities here; at
35-80 px tall, with the teams in one kit each, appearance does not tell
team-mates apart well enough.

    python train_player_reid.py --save-untrained .cache/player_reid_untrained.pt
    python identify_players.py output_st2_... --split \
        --reid .cache/player_reid_untrained.pt --cut-threshold 0.7 --look-threshold 0.85

## Events against the ball actions

The dataset's 21,428 ball actions say what happened and who did it (12
classes; in these windows three quarters are PASS and DRIVE, 1.4 s apart
at the median). `score_st2_actions.py` scores the pipeline's events
against them, per group, matched one to one within 0.5, 1 or 2 s, next to
the F1 that as many events at random times would get:

    pass   PASS, HIGH PASS, CROSS   <- pass
    carry  DRIVE                    <- carry
    won    TACKLE, BLOCK            <- tackle, recovery
    shot   SHOT, GOAL               <- shot, goal
    out    OUT                      <- out_of_play

**Pitch coordinates.** Goals, shots and out-of-play need positions on the
pitch, which the pipeline's pinhole ground plane cannot give for a
stitched panorama. The release's calibration does
(`soccertrack_v2.calibration`, `image_to_pitch`: fisheye undistortion,
then its homography): the ground-truth feet land a median 0.28-0.39 m
from their pitch positions on ten windows (0.78 m on M10, whose boxes are
projected). `run_pipeline(..., to_pitch=)` puts the tracks on the pitch
with it (`eval_soccertrack_v2.py --calibrated`). Two faults showed on
the way and are fixed for this path only (broadcast unchanged): the
chosen "ball" was a static object off the pitch in every window -- ball
candidates outside the pitch outline are now dropped -- and detecting
every second frame doubled every ball speed (`ball_time_aware`).

**Results** (detections and teams as above; nothing tuned on events):

| windows | group | labels | ours | F1 at 0.5 s | F1 at 1 s | F1 at 2 s |
|---|---|---|---|---|---|---|
| tuning (M3, M4, M6, M8) | pass | 89 | 24 | 0.12 | 0.16 | 0.23 |
| | carry | 71 | 33 | 0.23 | 0.37 | 0.40 |
| | shot | 3 | 23 | 0 | 0 | 0.08 |
| | out | 12 | 3 | 0 | 0.13 | 0.13 |
| report (M2, M10) | pass | 75 | 49 | 0.27 | 0.29 | 0.40 |
| | carry | 55 | 42 | 0.33 | 0.43 | 0.47 |
| | shot | 2 | 44 | 0 | 0 | 0 |
| | out | 9 | 0 | 0 | 0 | 0 |

Chance, per window at 1 s: passes 0.04-0.27, carries 0.09-0.25. Passes
are at chance on most windows (on M2 and M6 below it); carries are above
it on M3, M4 and M10 (p <= 0.011), marginally on M2 and M6 (p 0.06-0.07)
and not on M8 (p 0.48); shots are almost
all false (44 against 2 on the report windows); balls won and outs are
not found. Of the matched events, the team is right 25 times in 33 and
the actor 15 in 30 on the report windows.

**Why: there is almost no ball.** The actor is detected for 86-92% of
the actions, but the chosen ball is within 3 m of them for 11% (tuning)
and 22% (report) -- 4% on M3. The detector does not see this ball: on
M3's 72 action frames, with the release's ball track projected into the
picture, a detection of the COCO "sports ball" class (yolov8s at full
width) is within 60 px of it (the track is interpolated, and its point
sits 10-50 px off the ball) on 21% of them at confidence 0.02, and on 4%
at the pipeline's chosen 0.5. (An earlier count, 8% within 20 px, used a
radius smaller than the track's own error.) The ball is 5-6 px across
here.
The dataset has no ball boxes to train on (its ball track is interpolated
between the actions; its curated `mot/` boxes are players only). The
panorama ball detector below puts the ball at the actor for about half
the actions on the report windows.

## A ball detector for the panorama

No ball boxes are released, so the labels are weak, taken from the
actions: at a pass, dribble, cross, free kick or tackle, the ball is near
the actor's feet and near the release's ball track, and the calibration
puts both in the picture. Between actions the ball is somewhere near the
interpolated track.

- `build_ball_patches.py` streams a half and cuts 256 px patches around
  those points (jittered; `.cache/`, not committed).
- `build_ball_labels.py` gives each patch a code per 4 x 4 cell, sized in
  metres through the calibration (60 px is 3 m on the far side, 0.7 m on
  the near one):
  - **At an action:** a 1.5 m region around the actor's feet and the track
    point is "the ball is somewhere here". The other players' bodies and
    the actor's head are hard negatives. The rest is negative.
  - **Between actions:** a 2.5 m region, positive only.
  - **Ball-free patches** come from the same halves, 20 m or more from the
    ball.
- `src/ball_heatmap.py` is a ResNet-18 (ImageNet) cut at stride 8, with a
  stride-4 lateral and a one-channel head.
- `train_ball_heatmap.py` trains it with a multiple-instance loss: a smooth
  maximum over each region should be high, negatives low (focal). From
  step 1000 the worst 15-25% of bags in a batch are dropped, since a hidden
  ball makes a bag wrong.
- `src.ball_heatmap.detect` returns peaks above `tau` inside the pitch
  outline.

Training: first halves of M3, M4, M5, M6 and M8 gave 4,622 action bags,
1,944 between-action bags and 724 ball-free patches. M1 (another camera)
is excluded. 3,000 steps of 16 took 35 minutes on 4 CPUs.

**Protocol.** It was written before any checkpoint existed
(`eval_ball_detector.py`):

- **Tuning windows T:** second halves of M4, M6 and M8, minutes 10-13, so
  the same matches as training.
- **Report windows V:** M2 and M10, matches never trained on.
- **Primary measure, "free-flight hit":** over the middle half of every
  completed ground pass (10 m or more, team-mate receiver), the top
  candidate counts as a hit when it is within 1 m of the passer-receiver
  line and 2 m or more from every player. A detector of feet cannot pass
  this.
- **Baseline:** COCO's "sports ball" (yolov8s at full width, confidence
  0.02 and up, inside the outline).
- **Choices on T:** the checkpoint (best free-flight hit, the later
  within 0.02) and `tau` (the smallest giving at most one candidate a
  frame on random frames).
- **Gate G1 on T:** free-flight hit at least 0.30 and twice COCO's, before
  anything ran on V.

**On T the gate failed.** The checkpoint is step 3000 (0.240, against
0.234 and 0.203 for steps 2250 and 1500) and `tau` is 0.7.

| T (48 passes) | heatmap | COCO |
|---|---|---|
| free-flight hit (geometric) | 0.24 | 0.25 |
| difference, 95% interval | -0.01 [-0.10, 0.07] | |
| top candidate near the actor or track at PASS/DRIVE frames | 0.41-0.50 | 0.09-0.23 (random point: 0) |

**The gate's measure was wrong, and blind tiles show it**
(`ball_tiles.py`):

- **How the tiles work:** tiles around detectors' top candidates and
  around random pitch points (decoys) are shuffled and numbered. Each is
  judged "the match ball is in the marked 24 px box" or not (unsure counts
  as not). The key is read only afterwards. The judge is me, by eye.
  Over the four checks (410 tiles), decoys were judged "ball" 0 times in
  88. The judgements are not committed; `ball_tiles.py` redraws the tiles
  from `eval_ball_detector.py --save`.
- **The gate's own visual part passed:** of 40 confident top candidates
  (p >= 0.7) on T, 31 are the match ball. COCO's top candidates are the
  match ball 21 times in 40, plus 10 a ball resting by a goal.
- **What the tiles showed:** on one frame per pass in T, the heatmap's top
  candidate is the match ball in play 36 times in 48 and COCO's 28 times.
  Another 11 of COCO's are a ball resting by a goal, one of the heatmap's.
  Paired, the heatmap alone has 10 and COCO alone 2 (p = 0.04).
- **Why the geometric measure fails:** it missed 26 of the heatmap's 36
  confirmed hits and 15 of COCO's 28, and never counted a false one. A
  ball in flight is 1.3-3 m off the line between the two players'
  positions; 1 m was too tight.

So the run went on to V under a second protocol, written after T and
before anything ran on V. It keeps the same checkpoint and `tau`, adds the
blind free-flight check as primary, and still reports the registered
measures. That is a deviation, and it is reported as one.

**On V (M2 and M10, 38 passes):**

| V | heatmap | COCO |
|---|---|---|
| free-flight hit, geometric (registered) | 0.23 | 0.23 |
| difference, 95% interval | 0.00 [-0.07, 0.07] | |
| **free-flight, blind tiles: match ball in play** | **19/38 (0.50)** | **22/38 (0.58)** |
| passes found by one detector only | 3 | 6 (p = 0.51) |
| top candidate near the actor or track at PASS/DRIVE frames | 0.42, 0.43 | 0.15, 0.35 (random point: 0) |
| precision, blind: confident top candidate (p >= 0.7 / any) is the match ball | 32/40 (0.80) | 21/40 (0.53) |
| frames with such a candidate (random frames) | 0.64-0.80 | 0.50-0.64 |

**Events with the heatmap's candidates.** On M2 and M10, the heatmap's
candidates (`detect_ball_heatmap.py`, `tau` 0.7, at most 4 a frame, every
second frame) replace COCO's ball rows. Everything else is the cached run
that produced the events table above: the COCO run, redone the same way,
reproduces it exactly.

| M2 + M10, 1 s | COCO's ball | heatmap's ball |
|---|---|---|
| ball within 3 m of the actor at the action (M2, M10) | 0.11, 0.37 | 0.52, 0.55 |
| pass + carry: events, matched (130 labels) | 91, 39 | 282, 91 |
| **pass + carry F1, pooled** | **0.35** | **0.44** |
| difference, 95% interval (10 s blocks) | | +0.09 [-0.02, 0.21] |
| pass F1 / chance (M2; M10) | 0.19 / 0.26; 0.40 / 0.27 | 0.47 / 0.40; 0.42 / 0.32 |
| carry F1 / chance (M2; M10) | 0.35 / 0.21; 0.51 / 0.25 | 0.40 / 0.32; 0.48 / 0.29 |
| shots (3 labels) | 44 | 182 |

The ball is now at the actor for about half the actions instead of a
tenth to a third. The pipeline then fires three times as many passes and
carries. F1 rises by 0.09, but its interval includes zero, so no events
gain is claimed, as registered. Except for passes on M2, every F1 rose
less than its chance level (the F1 as many events would get at random
times) did, so most of the rise is the higher event count. Against chance
the picture is much as before: carries are above it on both windows (p
0.001-0.07), passes only on M10 (p = 0.03; COCO's 0.06). Shots, all
false, quadruple. The event rules were tuned
on a ball that was seldom there; with one that often is, they fire too
readily. That is the next thing to fix on the events side.

**What this says:**

- **What it does better:** the heatmap's confident pick is the ball four
  times in five where COCO's is half the time. 14 of COCO's 19 wrong
  picks are the penalty spots on M2. It also finds the play: at passes and
  dribbles its top candidate is near the actor two to three times as often
  as COCO's.
- **What it does not do better:** in free flight on matches it never saw,
  it finds the ball no more often than COCO (0.50 against 0.58).
- **For events:** with it, the ball is at the actor for half the actions,
  and pass and carry F1 is 0.44 against 0.35. That gain is not shown (the
  interval includes zero), and the event rules now overfire.
- **T flattered it:** its edge on T (0.75 against 0.58) was measured on
  the second halves of the matches it was trained on, and did not carry
  to new matches.
- **Why, plausibly:** the labels are mostly the ball at someone's feet,
  and only 724 ball-free patches taught it what is not the ball. At `tau`
  0.15 or below it gives the 8 candidates a frame it is capped at.
- **Caveats:**
  - one seed;
  - one judge (me), who knew the hypothesis but not the source of a tile;
  - 38 passes on V, enough only to see a large difference;
  - M2 and M10 are other matches but from the same rigs and league.

## Next

1. **The ball in flight.** The heatmap above finds the ball at the feet
   and is precise when confident, but in free flight on new matches it
   does no better than COCO. The labels are the next lever:
   - a bag along each ground pass's line (the ball is somewhere on it, for
     the pass's middle half);
   - far more ball-free patches, with the penalty spots and touchline
     balls as hard negatives;
   - all eight non-report matches.

   Then the measure should be the blind free-flight check on V, with the
   geometric measure's tolerance widened to the 3 m the tiles showed.
   Clicked labels (`make_ball_labeller.py`) remain the way to a proper
   test set.
2. A score that tolerates the boxes: the foot distance above, or the
   pitch positions once (1) exists.
3. **Teams, what is left**: giving every track its true majority side --
   the best any per-track rule can do -- reaches team accuracy 0.84 on
   M3, 0.88 on M10 and 0.82 on M2, because tracks switch from one player
   to another (see identities). The frozen rule is at that ceiling on M10
   (0.89; it refuses some impure tracks) and 0.04 below it on M2, where
   two kits of one hue are still close (the merge warning fires). Better
   teams now need tracks that stay on one player, not a better colour
   rule.
4. **Identities, what is left**: appearance is weak at these sizes (rank-1
   among team-mates 0.18-0.39 whatever the embedding, 0.20-0.29 for the
   untrained one used), so the next lever is
   where people are, not what they look like: on a fixed view, pieces of
   one player follow on in space and time, which `track_reid` and
   `join_by_look` use only loosely. With pitch coordinates (1), joins can
   be gated by running speed. A re-ID model would need far more distinct
   people than the 60 here -- the test matches M7 and M9, or other
   footage -- or larger crops.
