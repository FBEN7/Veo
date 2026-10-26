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

## Ball detector v2: the ball in flight, then events (protocol v3)

Registered here before any v2 training strip was cut, any v2 checkpoint
existed or any new report window was cut. The full protocol is kept
outside the repository with the judgements (sha256 5c5e7538...); what
binds is below.

**What changes from v1.**
- **Motion.** The network also sees motion: three grey channels next to
  the colour frame.
  - the differences to the frames 2 before and 2 after;
  - the difference to a running median background (15 frames, every
    50th, within ±14 s), which removes lines, spots and resting balls.
  - They feed a zero-initialised stem, so training starts from v1's
    ImageNet net (`src/motion_bg.py`, `build(motion=True)`).
- **Labels for the ball in flight** (`build_ball_strips.py`,
  `label_ball_strips.py`):
  - a corridor along every pass between passer and receiver, around where
    a decelerating ball should be;
  - every body outside the corridor as a hard negative;
  - ball-free strips with landmark hard negatives (penalty spots, centre
    spot, corners, posts);
  - a COCO "ball" found off the landmarks is ignored, not called "not a
    ball".
- **Data.** Both halves of M3, M4, M5 and M8, about 20,000 strips; the
  old T minutes ±60 s are kept out.
- **Loss.** A pool normalised to the ball's footprint, plus suppression
  of the strongest wrong peaks (`train_ball_heatmap.py --strips`).

**Splits.**
- **Choices:** every choice is made on M6 (U), a whole match never
  trained on: 4 windows.
- **Diagnostic only:** the old T windows (seen matches).
- **Reports:**
  - V-old is M2 and M10 as before.
  - V-new is 6 new 3-minute windows of M2 and M10, cut only after the
    detector is frozen: about 146 passes in all.
- **Never opened:** M7 and M9.

**Measure.** FF-auto replaces the 1 m rule. The 1 m rule is still
reported.
- **Rule:** a pass frame counts when the top candidate is:
  - within 4 m of the passer-receiver segment;
  - at the right place along it (|t - s| ≤ 0.30);
  - at least 1 m from every player.
- **Validation (`validate_ff_auto.py`):** against the 410 blind
  judgements above, kappa is 0.82 on T (where the rule was chosen) and
  0.81 on V-old. It never counts a decoy, and chance is 1.2%.
- **v1 by this measure:** 0.65 against COCO 0.50 on T, and 0.53 against
  0.50 on V-old.
- **Placebos:** the same rule just before the kick, and just after the
  reception. They catch a detector that has learnt "the grass between two
  team-mates".

**Decisions on U, in this order:**
1. the checkpoint (2500/3250/4000 steps);
2. a Platt calibration;
3. `tau`, the smallest giving at most one candidate a frame.

**Gate on U before anything v2 runs on V.**
- FF-auto ≥ 0.65 and ≥ COCO + 0.10, with the bootstrap lower bound above
  0.
- Placebos ≤ 0.05.
- A blind sitting: the match ball in play at ≥ 0.65 of U passes, the
  measure agreeing with the judge (kappa ≥ 0.6), and decoys ≤ 2 of 24.
- If the gate fails, one RGB-only fallback is trained. If that fails
  too, nothing runs on V.

**Claim on V (variant 2 of 3).**
- **H1:** v2 better than COCO by paired FF-auto over V-old and V-new.
  - 1-minute block bootstrap, 4,000 draws, rejected when the 2.5th
    percentile is above 0.
  - Paired SD 0.28, so the smallest detectable difference is about
    0.075.
  - The measure's validity is checked on new blind tiles. If that check
    fails, the blind McNemar is the primary.
- **"Clearly better"** also needs all of:
  - a difference of at least 0.10 by FF-auto and blind;
  - the same sign on M2 and on M10;
  - a positive difference on V-new alone.
- **H2:** v2 better than v1, tested only after H1.

**Events, after the detector is frozen and reported.**
- **Tuning:** a staged grid (`sweep_events_v2.py`) on two M6 windows.
  - It covers ball selection: the old one, or a linker in pitch metres
    that penalises a resting ball (`src/ball_link.py`).
  - It also covers the possession, carry and shot settings.
  - The objective is pass + carry F1 at 1 s above chance, with the event
    count kept within 0.6-1.5 × the labels.
- **Rules:** frozen, with their hash recorded, then run once on V-old.
- **Claims:**
  - H3: better than COCO's 0.35.
  - H4: better than v1's 0.44.
- V-old's events were seen when this grid was designed; that is
  disclosed.

**Deviations disclosed in advance:**
- the new measure;
- V extended;
- T demoted to a diagnostic;
- M6 held out;
- the label constants were chosen on T judgements pooled with V-old.
  The T-only figures are recorded: the ball sits a median 1.4 m off the
  line, and +0.14 of the pass ahead of the constant-speed point.

*Clarifications, written after the results:* where this summary differs
from the registered protocol file, the file binds.
- **V-new:** about 108 passes; 146 is with V-old.
- **FF-auto kappas:** they come from the 209 pass-frame detector tiles
  (112 T, 97 V-old), with the 88 decoys checked separately. V-old's kappa
  was seen during design.
- **G2:** the pre-kick placebo must be ≤ 0.05 and ≤ v1's + 0.03; the
  post-reception one ≤ max(v1, COCO) + 0.03.
- **G3:** it also needs PPV ≥ 0.85 and 9 of 12 confident picks.
- **Deviation:** the dropped U window st2_118578_1st_f015000 (9 passes)
  is a deviation too.

**Built and trained.**
- **Training data:** 21,340 strips from the 8 training halves: 7,004 in
  flight, 7,089 at actions, 5,015 ball-free, 1,432 dribbles and 800
  lofted passes.
- **Motion scales,** measured on the strips: D 0.0535 and F 0.0174.
- **Train against inference:** the motion input on a window decoded for
  inference matches the strips (D std ratio 0.97), so inference needs no
  change.
- **Training:** 4,000 steps, 84 min on 4 CPUs. The motion stem learnt
  (its weight norm grew from 0 to 0.99).
- **Held-out check:** of the 300 held-out strips, 218 have a ball region
  (actions, dribbles, flights). On those, the top peak falls in the
  ball's region 74% of the time (in flight 0.60).

**On U (M6, 63 passes) the gate failed, for both variants.** So, as
registered, the detector makes no claim on M2/M10, and V-new was never
cut. v2 did then run on V-old, for the events below.

| detector (M6) | FF-auto | minus COCO, 95% | placebo before kick / after reception |
|---|---|---|---|
| v2 with motion, steps 2500 / 3250 / **4000** | 0.603 / 0.599 / **0.627** | **+0.24 [+0.12, +0.37]** | 0.074 / 0.036 |
| the same, motion zeroed (diagnostic) | 0.575 | +0.19 [+0.10, +0.30] | 0.088 / 0.000 |
| v2 RGB-only fallback, 2500 / 3250 / 4000 | 0.440 / 0.560 / 0.540 | +0.15 [+0.03, +0.29] (4000) | 0.088 / 0.000 |
| v1 | 0.623 | +0.24 [+0.14, +0.35] | 0.074 / 0.018 |
| COCO | 0.385 | | 0.015 / 0.018 |

**COCO is depressed on M6 by resting balls by a goal**, as the protocol
registered: it scores 0.22 on M6's first half and 0.50 on its second. So
the margins over COCO are optimistic.

**The registered legacy measures favour v1.** Two of the four U windows
are in M6's first half, whose actions v1 was trained on.

| legacy measure (M6) | v2 | v1 | RGB-only | COCO |
|---|---|---|---|---|
| 1 m free-flight hit | 0.28 | 0.29 | 0.22 | 0.20 |
| action hit (194 actions; random point 0.005) | 0.34 | 0.43 | 0.33 | 0.10 |

**Why it failed:**
- **G1:** FF-auto 0.627 is below the 0.65 bar, although v2 is clearly
  above COCO.
- **G2:** the pre-kick placebo is 0.074, above 0.05. v1 has the same
  0.074, so this is not a corridor prior new to v2.
- **The fallback:** the RGB-only model fails both.
- **G3, the blind sitting,** was not held: with G1 and G2 failed, the
  gate could not pass.
- **Order:** Platt and `tau` were fitted after the gate decision,
  although the registered order put them before it. This does not touch
  G1 or G2, which use the top candidate at any p.

**What this says:**
- **The new labels did not make a better detector in this attempt** (one
  seed). The RGB-only model trained on them sits below v1: -0.08 [-0.15,
  -0.01] on U, and [-0.17, +0.02] on M6's second half.
- **Motion brings v2 back to v1's level by FF-auto, not beyond.**
  - v2 minus its own zeroed-motion run is +0.05 [-0.01, +0.10].
  - v2 minus v1 is +0.004 [-0.05, +0.06].
  - By the legacy measures v1 is ahead.
- **On M6's second half, unseen by both v1 and v2,** FF-auto is v2 0.64,
  v1 0.62 and COCO 0.50.

**Events go ahead with v2 at 4000 steps,** the highest on U by 0.004 over
v1. It is calibrated on U: Platt a = 0.895, b = -0.928, ECE 0.014, and
`tau` 0.25, giving 0.91 candidates a frame.
- **Why v2 ran on V-old:** the registered failure branch says events
  proceed with the variant of highest U FF-auto. That needs the variant
  on V-old, while the same clause also says "no v2 run on V".
- **How I read it:** as "no detector claim on V", so v2 ran on V-old for
  the events. V-old has now seen v2, its second of three variants; V-new
  is untouched.

**Event rules frozen before any V event** (`sweep_events_v2.py` on the
two M6 windows; events_v3.json sha256 fd260061...):
- **Ball selection:** the pitch-metre linker (`src/ball_link.py`; null
  cost 1, static penalty 2, top 4 candidates), replacing
  `select_single_ball`.
- **Possession radius:** 2 m (v1: 8 m).
- **Minimum pass distance:** 3 m (v1: 0.3 m).
- **Shots:** at least 50 km/h (v1: 10).
- **Everything else:** as v1.

On M6, where they were chosen (pass + carry at 1 s, pooled):

| | events (86 labels) | F1 | F1 at random times |
|---|---|---|---|
| frozen rules | 87 | 0.43 | 0.24 |
| v1 rules, same ball | 193 | 0.37 | 0.29 |

- **Gate G-E** (F1 above chance, pooled) passes: +0.19 against +0.08.
- **Leave one window out:** rules chosen on one window and scored on the
  other beat v1's rules on both windows (+0.12 against +0.01, and +0.19
  against +0.13).
- **Not registered:** in the sweep's first two stages no setting met the
  event-count constraint. The sweep then took the best objective
  regardless, a fallback the protocol does not define. The final
  re-check meets the constraint.

**Events on V (M2 + M10, the old windows).**
- **What ran:** v2 ran densely on these windows for this.
- **What was measured:** FF-auto and the other free-flight measures were
  not computed there. One registered legacy detector measure was: the
  events scorer's "ball within 3 m of the actor", below.

Pass + carry at 1 s, 130 labels:

| ball + rules | events | F1 | F1 at random times | F1 above chance | shots (2 labels) |
|---|---|---|---|---|---|
| C0: COCO + v1 rules (published) | 91 | 0.353 | 0.249 | +0.104 | 44 |
| C1: v1 heatmap + v1 rules (published) | 282 | 0.442 | 0.337 | +0.105 | 182 |
| C2: v2 + v1 rules | 264 | 0.497 | 0.340 | +0.157 | 180 |
| **C3: v2 + frozen rules** | **136** | **0.436** | 0.280 | **+0.156** | **13** |
| C0': COCO + frozen rules | 39 | 0.201 | 0.128 | +0.073 | 4 |

**The registered claims fail.**
- **H3** (C3 beats C0): +0.083, 95% interval [-0.053, +0.227]. Not shown.
- **H4** (C3 beats C1): not tested. It is -0.006 [-0.096, +0.086].

**What this says:**
- **The ball.** v2's ball is not shown to be better than v1's.
  - **Under v1's rules,** it gives F1 0.50 against v1's 0.44 (above
    chance +0.16 against +0.11), but that comparison was neither
    registered nor tested.
  - **Within 3 m of the actor:** v2's ball, through v1's selection, at
    0.51 and 0.57 of the actions (0.53 and 0.57 through the linker);
    v1's at 0.52 and 0.55.
- **The rules.**
  - **Counts:** the frozen rules make the event count match the labels
    (136 for 130, against 264-282).
  - **Shots:** false shots fall from about 180 to 13.
  - **Team:** at 0.5 s (the only tolerance at which the scorer checks
    the team), the team of the matched passes is right 20 times in 22,
    against 26 in 34 for v1's rules with the same ball.
  - **F1 above chance:** it stays at the same margin as v1's rules with
    the same ball (+0.156 against +0.157). So on V the rules gained
    nothing by the registered objective (on U: +0.19 against +0.08).
  - **Recall:** they match 58 labels against 98, so there are fewer
    false events but also fewer true ones.
- **COCO's ball under the new rules collapses** to 39 events. The new
  rules need a ball that is there, and COCO's is near the actor at only
  0.10 and 0.37 of the actions.

## Tracking the ball across frames (protocol v4)

Registered here before the tracker has run on any M6 test window or any
V-new window is cut. The full protocol is kept outside the repository
(sha256 f982b5fc...); what binds is below. Nothing is tuned: every value
comes from the frozen detector and the frozen events linker.

**Question.** Does linking v2's candidates over time, with short gap
fill, make v2's pick the match ball in play more often than v2's raw top
candidate at the same frame, in free flight, on matches never used for
any choice?

**The tracker (`src/ball_track.py`).**
1. **Input:** v2 dense at stride 2 (`detect_ball_heatmap.py --v2
   --stride 2`, no extra frames; anything else is refused). It keeps
   candidates with p_cal ≥ 0.25, the top 4 a frame.
2. **Link:** `ball_link.link` as frozen for events (c_null 1, c_static
   2), with no players, so no ground truth is used.
3. **Fill:** between two picks at most 6 frames apart, the ground point
   is interpolated in pitch metres and projected back, only when the
   implied speed is ≤ 35 m/s. No extrapolation.
4. **Fallback:** a detection frame with no pick and no fill gets the raw
   top-1.
5. **Frozen settings** (`tune_ball_track.py`, which only writes them):
   tau 0.25, top_k 4, c_null 1, c_static 2, gap 6, fill speed 35 m/s,
   fallback on (sha256 0ecf6095...).
6. **Where it runs:** inside `eval_ball_detector.py --track
   T=... --track-params ...`. Each pick records its source: detected,
   interpolated, fallback or none.

**Code frozen (sha256).**

| file | sha256 |
|---|---|
| `src/ball_track.py` | 68e287a4... |
| `src/ball_link.py` | 4250c6c2... (as in events v3) |
| `eval_ball_detector.py` | a98e7bc5... |
| `ball_tiles.py` | 1715eabf... |
| `score_ball_track.py` | 9c0bdf4d... |
| `validate_ff_auto.py` | d2cebb59... |
| checkpoint `ball_v2.4000.pt` / calibration | 31f9a5eb... / 77799641... |

**Splits.**
- **Dry run, U-E** (the 2 M6 windows events were tuned on, 24 passes):
  counts and implementation checks only.
- **Gate, U-test** (the other 2 M6 windows, 39 passes).
- **Report, V-new:** the 6 M2 / M10 windows registered in protocol v3,
  still not cut, about 108 passes.
- **Not used:** V-old. M7 and M9 are never opened.

**Measure: blind pair tiles.** FF-auto rewards any point on the pass
line, which is where interpolation puts the ball, so the judge is
primary.
- **Pair frames:** every pass frame (q0-q3) where the tracked and raw v2
  picks are more than 6 px apart or exactly one is empty, plus q1 of
  every pass.
- **Judging:** a tile counts when the centre of the match ball in play is
  inside its 24 px box; unsure counts as not. The judgements are hashed
  before the key is read (`score_ball_track.py` refuses otherwise).
- **Decoys:** random, corridor, and offset (a tracked pick moved 18-20 px,
  away from every candidate), plus re-judged old tiles.

**Gate on U-test** (all must hold, else U-test is reported and V-new is
not cut):
- **G-a:** passes where tracking gains outnumber passes where it loses.
- **G-b:** tracked placebo hits ≤ raw's + 1, before the kick and after
  the reception.
- **G-c:** decoys judged the ball ≤ 2 of 20 (random + corridor) and ≤ 1
  of 6 (offset).
- **G-d:** stride-only input, the registered checkpoint and calibration,
  no fill over 6 frames, and no detected or fallback pick at an odd
  frame.

**Claims on V-new** (one shot, sealed until judged):
- **H5 (primary):** per pass, d = frames where only the tracked pick is
  the ball minus frames where only the raw one is. Exact one-sided sign
  test over passes with d ≠ 0, p < 0.025, with more d > 0. The expected
  size is about 63 informative passes, power about 0.75 at a 70/30
  split.
- **H6 (only if H5 holds):** tracked v2 against COCO's raw top-1 at q1,
  exact one-sided McNemar.
- **Secondary:**
  - the q1-only McNemar;
  - a breakdown by pick source, frame parity, match and distance;
  - FF-auto paired, counted only if it agrees with the judge on
    interpolated tiles (kappa ≥ 0.6);
  - placebos and decoys.

**Disclosed.**
- One seed, one detector.
- The Platt fit used M6 pass frames, so U numbers are optimistic; V-new is
  unaffected.
- The linker constants were chosen on U-E for events, with players; here
  it runs without them.
- GT-free counts on U-E were seen before this registration.
- At odd frames raw has a detection the deployed tracker never ran,
  which is conservative for the tracker.
- V-new is the last unseen M2 / M10 data, and this spends it.

**Gate on U-test: passed** (recorded before any V-new window was cut).
- **Run:** the tracker covered every measured frame. No fill was longer
  than 6 frames, and the dense top-1 equals the evaluation's top-1 at all
  228 shared frames.
- **Sitting S-U:** 122 tiles judged blind (judgements sha256 228cd536...
  recorded before the key was read): B 60, N 58, a ball off the pitch 3,
  a ball resting by a goal 1.

| | result | bar |
|---|---|---|
| G-a: passes where tracking gains / loses | 11 / 3 (sign test p = 0.029) | gains > losses |
| G-b: placebo hits, tracked / raw | before the kick 3 / 4, after 1 / 1 | ≤ raw + 1 |
| G-c: decoys judged the ball | 0 of 20; offset 0 of 6 | ≤ 2; ≤ 1 |
| G-d: code check | stride 2, registered checkpoint and calibration, fills ≤ 6 | all |

- **Where the change comes from:** 39 passes give 66 pair frames.
  - **Gains:** all 13 frames where only the tracked pick is the ball are
    interpolated: 6 detection frames the linker left empty, and 7 odd
    frames.
  - **Losses:** all 3 frames where only the raw pick is are odd frames
    with no tracked pick.
  - **At detection frames** the two agree (14 both right, 1 both wrong).
  - **At q1 alone** the comparison is 1 against 1, which is why pair
    frames are the primary.
- **Doubtful tiles:** counting every doubtful raw-only tile as the ball
  (three were a ball off the pitch, by the fence) still gives 9 against
  5.
- **Re-judged old tiles:** 6 of 6 agree.
- **FF-auto is not valid for tracked picks.** On interpolated tracked
  tiles it agrees with the judge at kappa 0.17 (PPV 0.89), so for V-new it
  stays a secondary measure, reported as invalid, as registered.
- **The registered expected size was too optimistic.** On U-test only 36%
  of passes are informative (d ≠ 0), so V-new should give about 39 such
  passes, not 63. Power is about 0.62 at a 70/30 split; U-test's 79/21
  would give 0.94.
- **U numbers are optimistic:** the Platt fit used M6.

**Result on V-new: the registered claim is not met.** The 6 new windows
of M2 and M10 hold 139 passes.
- **Run:** every measured frame was covered. The dense top-1 equals the
  evaluation's top-1 at all 754 shared frames, no fill was longer than 6
  frames, and the code check holds.
- **Sitting S-V:** 332 tiles judged blind (judgements sha256 48219b60...
  recorded before the key was read): B 147, N 156, a ball off the pitch or
  not in play 25, a ball in or by a goal 4.
  - Decoys judged the ball: 0 of 24 random and corridor, 0 of 12 offset.
  - Old tiles re-judged: 12 of 12 agree.

| | tracked only | raw only | one-sided p | bar |
|---|---|---|---|---|
| **H5: passes (pair frames)** | **18** | **9** | **0.061** | < 0.025 |
| q1 alone (McNemar) | 7 | 2 | 0.090 | secondary |
| H6: tracked v2 vs COCO at q1 | 37 | 2 | -- | not tested: H5 failed |

- **Underpowered.** Only 27 of 139 passes are informative (d ≠ 0), against
  a registered expectation of 63 (U-test suggested 39). With 27 the test
  needs 20 passes for tracking, and power at a 70/30 split is 0.41.
- **Where it comes from** (descriptive, not tested):
  - **At detection frames (even),** the tracked pick alone is the ball 14
    times and the raw pick alone 0 times:
    - 3 times the linker took a lower-ranked candidate that was the ball;
    - 11 times it rejected a wrong top-1 and a fill put the pick on the
      ball.
  - **At odd frames,** it is 6 against 10. Nine of raw's 10 are odd frames
    where the tracker has no pick: the gap was over 6 frames, or too fast
    to fill. Raw has a detection there only because the evaluation detects
    at every measured frame; the deployed stride-2 tracker never does
    (disclosed at registration as conservative for the tracker).
  - **By match,** at frame level: M2 10 against 3, M10 10 against 7.
- **Robustness to the judge** (not registered).
  - **Three further blind judges** (no access to the key) labelled the
    same 332 tiles. Each agrees with the registered judgements on ball /
    not ball at 324-325 of 332 tiles.
  - **Their H5:** 17 / 9 (p 0.084), 18 / 11 (p 0.13) and 18 / 9 (p
    0.061). The majority of the three gives 19 / 9 (p 0.044), and the
    majority of all four 17 / 9 (p 0.084).
  - **Bounds:** if every tile I marked doubtful went against tracking it
    would be 13 / 11; if every one went for it, 22 / 7.
  - Every judge gives the same direction, and no judging reaches the
    registered p < 0.025.
  - An independent recomputation of every count from the judgements and
    the key matched the scorer.
- **FF-auto:** on interpolated tracked tiles it agrees with the judge at
  kappa -0.00. So its paired difference (+0.031, interval [+0.005,
  +0.053]) does not count as a measure, as registered.
- **Placebos:** before the kick, tracked 9 against raw 10 (136 frames);
  after the reception, 8 against 9 (128 frames).
- **The scorer prints "G-c FAIL" on S-V.** That gate checks M6's
  registered decoy counts (20 and 6), and S-V used 12 per kind. No decoy
  was judged the ball.

**What this says.**
- **Against raw v2:** tracking is not shown to put v2's pick on the ball
  more often at the same frame. The direction favours it in both matches
  and for every judge, but the effect is rare: about 1 pass in 5 changes,
  and about 2 in 3 of those favour tracking.
- **Against COCO:** the tracked pick looks far better at q1 (37 against
  2). The registered order stops at H5, so this is untested.
- **What would remove the losses** (post hoc): they are all frames the
  stride-2 tracker never ran. Tracking at stride 1, or filling longer
  gaps, would remove most of them. That is a hypothesis for new data, not
  a result.
- **V-new is spent.** A further test needs the test matches (M7, M9) or
  clicked labels.

## Verified ball labels (step 1 of the ball plan)

The aim is labels of where the ball is, judged rather than derived from the
actions, to train a v3 detector and to test on M7 / M9. Files are in
`.cache/soccertrack_v2/ball_verified/` (`A`: M6, `B`: the 14 other seen
windows, `clf`: the tile classifier).

- **Candidates** (`build_ball_verify.py`): at the measured frames of each
  seen window, the positions proposed by v2 (top 2), v1, COCO, the tracker
  and the pass line, merged within 6 px. Each is a tile with a 24 px box.
- **Judging** (`label_ball_verify.py`, `spot_check_tiles.py`): blind judges
  label each tile B (the match ball in play, centre in the box), B* (a ball
  in or by a goal), S (a ball off the pitch or not in play) or N.
  - **M6:** 2,403 tiles, two judges and a third on disagreements. The judges
    agree on 97.4% (kappa 0.935), and judged no decoy the ball (0 of 56).
    546 of 762 frames have a verified ball.
- **A classifier in place of the judges** (`train_tile_classifier.py`): a
  small CNN on the 64 px crop, its motion against t±2 and t±50, which
  sources proposed it and where it is on the pitch. A tile is labelled
  "B" at p ≥ 0.95 when it is its frame's top tile, "not" at p ≤ 0.1 or when
  it is away from the top tile, and is left to a judge otherwise.
- **Trained on M6 alone, it did not transfer.**
  - **Spot check (100 tiles):** 46 / 50 of its "B" were the ball, and 4 / 50
    of its "not".
  - **300 random tiles of the 6 other matches, judged:** 39% were the ball.
    Its "B" caught 26% of them, and 22% of its "not" were the ball
    (weighted by window). The 50-tile spot check had underestimated this.
  - **Why:** the tracker was not run for 4 of those matches, and on M6 most
    balls carried its flag. 3 matches are also darker than M6.
- **Retrained** on M6 plus 500 judged tiles of the 6 other matches.
  - **The 500 tiles:** the spot check, plus 300 random and 100 the classifier
    was unsure of. One judge labelled each tile; a second rechecked every
    tile not labelled N, the unsure ones and 10% of the rest (250 of 255
    agree), and a third decided the 5 disagreements.
  - **Settings:** the tracker flag is dropped (it depends on which steps ran
    on a window), and brightness augmentation is widened to 0.6-1.4.
  - **One match held out at a time, on the random tiles:** agreement 0.89.
    At p ≥ 0.95, 95% are the ball; at p ≤ 0.1, 6% are; 81% of tiles fall
    in one of the two. Keeping the old settings gives the same within noise.
  - **With M6 held out** (trained on the 500 tiles alone): precision 0.55.
    Little data from other matches does not transfer.
- **Auto-labels of the 14 windows** (`B/auto_labels_AB.parquet`): of 8,068
  tiles, 2,525 are "B", 4,905 "not" and 638 left to a judge. 63% of
  frames have an auto ball (M6, judged: 72%).
- **Spot check of 100 fresh tiles** (two judges, a third on 3
  disagreements): 50 / 50 of "B" are the ball, and 3 / 50 of "not".
  - These windows are now training matches, so this measures label
    quality, not transfer.
  - 50 tiles of "not" are too few: the held-out figure (6%) is the better
    estimate.
- **Cost:** about 2M agent tokens for the judging of the 600 tiles of
  the other matches.

## Ball detector v3 (protocol v5, part 1)

v2's architecture fine-tuned on the verified labels. How the checkpoint
would be chosen was committed before any training
(`ball_verified/v3_selection.json`, with amendment 1). The claim is left
to M7 / M9 (part 2).

- **Data** (`build_verified_strips.py`): the 14 B windows of M2, M3, M4,
  M5, M8 and M10, cut from the window videos.
  - **Ball strips:** 3 crops per frame with a located ball, 5,815 in all.
    The ball's 24 px box is a footprint bag with a 0 ring to 40 px.
  - **Negative strips:** 861 crops on judged N tiles, or on confident
    "not" tiles when the frame's ball is elsewhere.
  - **Codes:** -3 on judged N boxes, 0 near spare, static, COCO-proposed
    or unsure tiles, and -1 elsewhere only when the ball is located.
  - **v2's stream strips:** 20,738, less the 602 inside B windows.
  - **Refused:** M6, M1, M7 and M9.
- **Training** (`train_ball_v3.py`):
  - **Two runs:** v3a is fine-tuned from v2.4000; v3b from v1 (motion
    stem at zero).
  - **Schedule:** 2,000 steps on CPU (about 42 minutes each).
  - **Batches:** 8 ball, 3 negative and 5 of v2's strips.
  - **Weights:** an average of the weights is saved at 1,000, 1,500 and
    2,000 steps, with the early batch norms frozen.
- **Selection set:** M6's second half (2 windows, 320 ball frames).
  - v1 was trained on M6's first half, so amendment 1 dropped those
    windows before any training.
  - Measure: the top-1 within 12 px of the verified ball.
  - Guard: the FF-auto placebos at most v2's + 0.03.
  - Rule: the highest score, or a later step of the same run within 0.02.

| checkpoint | top-1 is the ball | ball in top 4 | placebo guard |
|---|---|---|---|
| v3a.1000 / 1500 / 2000 | 0.919 / 0.906 / 0.903 | 0.978 / 0.978 / 0.975 | pass / pass / **fail** (3 / 32 after reception) |
| v3b.1000 / **1500** / 2000 | 0.931 / **0.947** / 0.925 | 0.991 / 0.994 / 0.994 | pass |
| v2.4000 | 0.819 | 0.950 | reference |
| v1 | 0.856 | 0.966 | |

- **Chosen: `ball_v3b.1500.pt`** (`v3_decision.json`).
  - **Against v2.4000:** +0.128 [+0.095, +0.172]. It is right on 43 frames
    where v2 is wrong, and wrong on 2 where v2 is right.
  - **Balls v2 did not propose** (32 frames): v3 finds 0.72, v2 0.
  - **v2.4000 re-run in the same jobs** reproduces its saved picks on all
    427 frames, so the evaluation itself did not change.
- **Frames with no verified ball** (105):
  - v3's top-1 lands on a judged non-ball 62 times and elsewhere 43 times.
  - Those 43 were judged blind, mixed with 40 controls. The judges agree
    with the controls on 39 of 40.
  - Only 3 of the 43 are a ball nobody had proposed. 10 are the penalty
    spot, a new failure: when no ball is visible, v3 picks the spot.
  - v2's top-1 is wrong on all 105 of these frames, v3's on 102.
- **Why this is a diagnostic, not a claim:**
  - v3 is the best of 6 checkpoints on this very measure.
  - The B labels come from a classifier that learned mostly from M6
    judgements.
  - The verified positions are v2's peaks where v2 proposed the ball,
    which favours v2.
  - v2's checkpoint and calibration were also chosen on M6.
- **Next:** v3's calibration and tau on M6, as registered. Then stride-1
  tracking with v3, protocol v5 part 2 registered, and the test on
  M7 / M9 with fresh blind tiles that have v3 as a source. A penalty-spot
  negative (v2's landmark codes) is a candidate fix, but it would need a
  new, logged run.

### Step 3: calibration and tracking with v3

Registered before any dense v3 run (`v3_track_selection.json`).

- **Calibration:** `calibrate_ball_heatmap.py` as for v2, on M6's second
  half: a 0.905, b -2.068, ECE 0.024. Tau is 0.10 by the registered rule
  (at most one candidate a frame on random frames; v2's is 0.25).
- **Dense v3:** every frame of M6's two second-half windows (9,000
  frames, about 2 hours on CPU in 750-frame chunks).
- **Tracker:** the v4-r1 tracker as frozen (`src/ball_track.py`: top 4,
  c_null 1, c_static 2, gap 6, fill speed 35 m/s, fallback), with v3's
  tau, at stride 1. The gap of 6 frames stays valid there (< 7).
- **Rule:** keep tracked v3 for M7 / M9 if its top-1 hit is within 0.01
  of raw v3's and the placebos pass the guard.

| M6 second half, 320 ball frames | top-1 is the ball | balls v2 did not propose (32) | placebo pre / post |
|---|---|---|---|
| **v3 tracked, stride 1** | **0.975** | 0.91 | 0.10 / 0.03 |
| v3 tracked, stride 2 | 0.956 | 0.91 | 0.10 / 0.03 |
| v3 raw | 0.947 | 0.72 | 0.10 / 0.03 |
| v2 tracked, stride 2 (v4-r1) | 0.847 | 0.41 | 0.08 / 0.06 |
| v2 raw | 0.819 | 0 | 0.10 / 0.06 |

- **Chosen for protocol v5 part 2:** v3 tracked at stride 1
  (`v3_track_decision.json`).
- **Against raw v3:** +0.028 [+0.013, +0.048]. Tracking fixes 9 frames and
  breaks none.
- **Stride 1 against stride 2:** at stride 2 the odd frames are
  interpolated or missed (47 misses on the measured frames).
- **What tracking does not fix:** 9 of raw v3's 10 penalty-spot picks
  on frames with no visible ball are still there. With no linked pick,
  the fallback takes the raw top-1. It keeps the 3 balls only v3 found.
- **Same caveats as step 2:** all of this is on M6, which was used for
  every choice, so it is a diagnostic. The claim is for M7 / M9.

### Step 3b: the painted spots (`src/ball_spots.py`)

The view is fixed, so the centre spot and the two penalty spots never
move in the image. Each attempt was registered before it touched M6
(`v3_spot_selection.json`). The rule: keep the filter if the tracked
top-1 hit stays within 0.01 and the placebos pass.

- **First attempt (registered, failed):** drop candidates within 0.75 m
  of a spot as the calibration places it.
  - The radius came from the 14 training windows: all 212 candidates
    within 0.5 m of a spot are static, and none was judged the ball.
  - On M6 it changed nothing. M6's calibration puts the far penalty spot
    about 25 px (some 7 m on the far side) from the painted one, where
    v3 fires. This was found by looking at M6's spot picks.
- **Amendment 1 (kept):** find the painted spot from the window's own
  candidates, with no labels, so it works the same at test time.
  - Search within 60 px of each projected spot.
  - A point is a mark when it holds a candidate on at least 5% of the
    frames (9 s of 3 minutes; a ball in play never rests that long).
  - Drop the candidates within 4 px of a mark.
  - The constants come from the setting, not from M6's numbers.
  - On M6 it finds the far penalty spot at (3099, 293) and the near one
    (plus a second persistent mark near it).

| M6 second half | tracked v3 | tracked v3, marks dropped |
|---|---|---|
| top-1 is the ball (320 ball frames) | 0.975 | 0.975 (no frame lost) |
| picks on the penalty spot (the blind check's 10) | 9 | **0** |
| balls only v3 found (3) | kept | kept |
| placebo pre / post | 0.10 / 0.03 | 0.10 / 0.03 |

- **Kept for protocol v5 part 2** (`v3_spot_decision.json`): v3b.1500,
  dense at stride 1, marks dropped, tracked (v4-r1, tau 0.10).
- **Still open:** the tracker's fallback always gives a position. On the
  10 spot frames the pick moves elsewhere: 2 more on judged non-balls,
  the rest unverified. Whether a frame should get no position when
  nothing is credible is a separate design question.

## Protocol v5 part 2: the test on M7 / M9 (registered)

Registered in `ball_verified/v5p2_registration.json` before any content of
M7 or M9 was read. So far only their release files' sizes have been read.
The registration holds the sha256 of every file used. `src/unseal.py` opens
a test match only when given the registration's sha256 and while every
frozen file is unchanged, and it logs each opening.

- **Windows:** M7's first half and M9's second half, 3 minutes from
  minute 20 of each, about 300 ball frames (the budget chosen).
- **Compared:**
  - v3P: v3 at every frame, painted-spot marks dropped, tracked at tau
    0.10.
  - Against v2's raw top-1 (H1) and, only if H1 holds, v1's (H2).
  - v3 raw and COCO are described only.
- **Ground truth:** fresh blind tiles from the five detectors' top-1,
  merged with the tile at the mean of its sources, so no detector's peak
  is the reference.
  - One judge reads every tile. A second reads every tile not labelled N,
    the unsure ones and 1 in 7 of the rest. A third decides
    disagreements.
  - The tile classifier is never used on test tiles.
- **Test:** the top-1 within 12 px of the verified ball, on ball frames.
  - A one-sided sign-flip test over 10-second blocks, at p < 0.025.
  - The frame-level McNemar is reported alongside.
  - A null result is reported as such.
- **Power:** a gain of about 8 points would be detected; one of 4 points
  only about half the time.

### Result: H1 not rejected (v3 not shown better than v2 on M7 / M9)

Run once, as registered, with amendments 1-2 (the CPU part in
restart-sized jobs; dense v3 in 250-frame chunks, rows unchanged).
Judging ran before the user's weekly usage reset, at the user's request
(amendment 3: timing only).

- **Labels** (`v5p2/labels`): 801 judged tiles, 357 measured frames.
  - 188 frames have a verified ball (M7 76, M9 112), 168 are unknown and
    1 is a conflict. That is fewer ball frames than the ~300 planned.
  - The judges agree on B / not B on 0.99 of tiles (kappa 0.98). No
    decoy was judged B. On the random recheck, 52 of 53 agree.
- **H1** (`v5p2/scores_H1.json`): v3P is right on 0.734 of the ball
  frames, v2 on 0.585 (+14.9 points).
  - v3P alone is right on 54 frames, v2 alone on 26.
  - Sign-flip p = 0.062 over 25 blocks, so H1 is not rejected at 0.025.
    The frame-level McNemar (p = 0.001) is reported only; it ignores the
    blocks.
  - H2 (v1) is therefore not tested.

| Top-1 is the ball | all (188) | M7 (76) | M9 (112) | hit@4 |
|---|---|---|---|---|
| v3P | 0.734 | 0.934 | 0.598 | -- |
| v3 raw | 0.702 | 0.882 | 0.580 | 0.947 |
| v2 raw (C1) | 0.585 | 0.658 | 0.536 | 0.910 |
| v1 raw | 0.325 | 0.368 | 0.295 | 0.681 |
| COCO | 0.426 | 0.382 | 0.455 | 0.431 |

- **Placebos:** no detector picks the ball before the kick or after the
  reception, except v1 after the reception (0.04).
- **What went wrong on M9:**
  - Its view has balls resting in or by a goal: 168 tiles were labelled
    B*, against none on M7.
  - On 92 of M9's 99 unknown frames, the only ball among the tiles is one
    of those. Every detector's top-1 was on a judged tile there, never on
    the match ball.
  - On M9's ball frames all detectors are at 0.54-0.60. The
    tracker's static-candidate cost did not keep v3P off these balls.
  - The gain is concentrated on M7 (+28 points), which is why the
    block test, with 25 blocks, does not reach 0.025.
- **What it means:** v3P is the best pick measured on both matches,
  with a large gain on M7. But the registered test does not show it
  better than v2. M7 and M9 have now been used, so any change made
  after seeing these results needs a new test set.

### Events on M7 / M9 with the v3 ball (descriptive, after the test)

Not registered; run after H1 was scored (`research/session_scratchpad/ball/events_m7m9/`).
- **Setup:** players and teams as for the old windows (`eval_soccertrack_v2.py
  --run`, stride 2, then `events_run.py`). The events_v3 rules are frozen,
  with each ball's own calibrated tau (v2 0.25 as frozen; v3 0.10). The
  rules' linker takes the top 4, not v3P's tracker.
- **Scored:** pass + carry against the 107 labels of the two windows,
  F1 at 1 s pooled.

| ball + rules | events | F1 | chance | above chance | M7 above | M9 above |
|---|---|---|---|---|---|---|
| COCO + v1 rules (events_v1) | 55 | 0.321 | 0.180 | +0.141 | +0.180 | +0.115 |
| v2 + frozen rules | 80 | 0.310 | 0.195 | +0.116 | +0.148 | +0.070 |
| v3 + frozen rules | 85 | 0.323 | 0.202 | +0.121 | +0.182 | +0.044 |
| v3 without the spot marks + frozen rules | 86 | 0.321 | 0.203 | +0.118 | +0.174 | +0.047 |

- **Paired:** v3 minus v2 is +0.013, 95% [-0.092, +0.126], over 33 blocks.
  The events do not improve measurably.
- **Why so little, probably:**
  - The events linker reads the top 4, where v2 already holds the ball
    0.91 of the time (v3 0.95). v3's large top-1 gain mostly does not
    reach the events.
  - On M9 every ball does poorly: balls are at rest by the goals, and
    the camera is different (3840 x 1504).
  - 107 labels give wide intervals.
- **So:** a better top-1 ball is not what limits the events now. The
  next lever is on the events side (possession, the linker on a stride-1
  ball), measured on unseen footage.

### Events plan, steps 1-2: the ball for the rules, the rules retuned, tested on M1

**Where labels were lost** (M7 + M9, `events_m7m9/funnel.py`, 110 pass / carry /
shot labels):
- the actor was detected for 0.70;
- some v3 top-4 candidate was within 3 m of the action for 0.52;
- the ball the rules kept was there for only 0.24;
- with both the actor and the ball there, an event fired for 2 of 22.

**Step 1 (the tracked v3 ball for the rules):** at most +0.01 above chance
on M7 / M9. With the tracker's fallback, there were 80 false shots on M7.
On M9 the tracker sat on a ball resting by the goal.

**Step 2a (`events_tune/`):**
- The ball input and the rules were retuned on M6's second half, M7 and M9
  (160 pass + carry labels).
- New ball input: static points dropped anywhere. A point holding a
  candidate on 5% of frames goes, which removed none of the 188 verified
  balls.
- The best setup (E2) was v3 with static points dropped, the linker, a
  possession radius of 3 m and a hold of 0.4 s. It scored +0.223 above
  chance against +0.163 for the current pipeline (E0).
- With each window held out in turn, E2 beat E0 on all three.

**The M1 test** (registered in `events_tune/m1_registration.json` before any M1
output was read; 4 windows of M1, never used by any ball model or rule;
365 pass + carry labels; one amendment, resumable player detection,
result-neutral):

| setup | events | F1 at 1 s | chance | above chance |
|---|---|---|---|---|
| E0 current (v2, linker, frozen rules) | 264 | 0.420 | 0.323 | +0.097 |
| E1 v3 tracked + retuned rules | 284 | 0.450 | 0.342 | +0.108 |
| E2 v3 static-filtered + linker + retuned rules | 310 | 0.468 | 0.360 | +0.109 |

- **Primary: E2 minus E0** is +0.048 in F1, 95% [-0.020, +0.114] over 72
  blocks. Not shown. E1 minus E0 is +0.030 [-0.030, +0.091].
- **Most of the F1 gain is more events:** chance rises with the event
  count (0.323 to 0.360), and above chance E2 gains only +0.011.
  - E2 is above E0 on three windows (+0.20 / +0.09 / +0.13 against
    +0.13 / +0.05 / +0.09 above chance).
  - E2 falls to chance on the fourth, 2nd f027000 (+0.01 against +0.11).
- **So the dev gain did not carry over.** Three windows of tuning overfit;
  the leave-one-window-out check could not catch that the windows share
  one tuning set.
- Retuning a handful of hand rules is not the lever. Learned events, or
  a much larger tuning set, are what is left.

### M1's camera refitted, the test rerun (amendment 2; exploratory)

**Corrected after an adversarial review.** The first version of this section
said that M1's calibration was broken and that the other releases were
imperfect too. Both claims were wrong. The checks below are the corrected
reading.

**The fault is a frame mismatch, not a bad calibration.**
- The release calibration agrees with the release's own labels. The truth's
  box feet, projected, land a median 0.36-0.41 m from the truth's positions
  on all four M1 windows, which is the same as on other matches.
- M1's labels, boxes and homography alike, sit in a different pixel frame
  from the video:
  - 99% of the truth's box feet are at v <= 960-1040;
  - our detections reach v ~1510-1670.
- No vertical affine maps the one onto the other: the best still leaves a
  median of 42-63 px.
- Our detections' feet, projected with the release, sit a median 1.8-1.9 m
  from the truth's positions; only 0.24-0.32 of them fall within 3 m.

**The refit (`calib_fix/fit_players.py`) is a warp, not a camera.**
- What it fits: the inverse homography and K's vertical intrinsics, against
  the truth's positions on 1st f001500 and 2nd f000000.
- What it does: it brings our detections' feet to a median 0.64-0.67 m on
  the two other windows. One-to-one matching gives the same median (about
  0.9 m near the goals).
- Why it is not a camera:
  - its parameters are non-physical: fy goes from 946 to 70,026;
  - the homography's u column shrinks to 1-7%;
  - it maps our player boxes' bottoms, foot-point bias included, and
    nothing else.
- **Do not use it for ball, shot or goal geometry.**
- A physical fix would fit the camera to the release's pitch-line outline,
  which is in the video's frame (`src/soccertrack_v2.py`).

**The other releases are fine; our foot point is not** (`.cache/calib_diag/`
holds refits of M6/M7/M9, kept only as evidence; do not use them).
- The release projects the truth's box feet to 0.30-0.50 m on M6, M7 and M9.
- Our detections' feet sit 0.9-1.4 m off. The difference is our detector's
  foot point:
  - our boxes are taller (median crop_h 43-51 px against the truth's 40-45);
  - they sit a median ~20 px from the truth's feet in the image, which is
    metres at the panorama's ends.
- The refits there make the truth's feet worse: 2.55 m on M6's 2nd half,
  and 13.7 m near the goals.
- **So position accuracy is limited by the detector's foot point.** The next
  lever is a foot-point correction learned against the truth's boxes on the
  training matches, then re-measured.

**The M1 events test rerun** under the refit (same frozen code and rules,
`m1_cf_*.json`). This is exploratory:
- the amendment came about 5 h after the registered result had been read;
- it was fitted on the player truth of two of the four test windows.

The registered primary stays the release run: **not shown**.

| setup | F1 at 1 s, release → refit | above chance, release → refit |
|---|---|---|
| E0 | 0.420 → 0.441 | +0.097 → +0.111 |
| E1 | 0.450 → 0.460 | +0.108 → +0.120 |
| E2 | 0.468 → 0.474 | +0.109 → +0.111 |

- E2 minus E0 is +0.033, 95% [-0.024, +0.090]; E1 minus E0 is +0.019
  [-0.038, +0.074]. Same reading: not shown.
- The positions were worth about +0.01-0.02 in F1 on every setup. They are
  not what holds events back.

### Learned events: shown better than the rules on M1 (`events_learn/`)

**The model (`learn.py`):** per detection frame, features of the ball and
of the players around it, with offsets of up to ±12 frames. One
gradient-boosted classifier per class answers "a pass / a carry starts
within 0.24 s". Events are the peaks of its smoothed probability, at least
0.6 s apart.
- The threshold per class is set by event rate: it is chosen so that the
  training matches' own out-of-fold events number as many as their labels.
- An earlier choice maximised F1 on the scored folds themselves. It was
  optimistic (+0.240 against +0.205 on three windows), so it was dropped.

**Training:** 9 windows of M2, M3, M4, M6 (both halves), M7, M8, M9 and M10,
450 pass + carry labels. The ball input is v3, static points dropped, linked.

**Cross-validation:** nested, leaving one match out. Pooled F1 at 1 s is
0.673 against a chance of 0.269, or **+0.405** above chance.
- On the windows whose ball v3 never trained on: M7 +0.43, M9 +0.20.
- On M6 2nd, which was in v3's selection set: +0.48.
- The other windows come from matches v3 trained on, so their ball is
  better than on new footage.

**The M1 test** (`m1_learned_registration.json`, registered before any M1
feature was computed):
- This is a second use of M1, on the owner's choice; it is weaker
  evidence than fresh footage.
- It runs under M1's refit, like the E0 rerun it is compared with.
- It was scored once.

| setup | events | F1 at 1 s | chance | above chance |
|---|---|---|---|---|
| E0 rules (current) | 288 | 0.441 | 0.330 | +0.111 |
| E2 rules (retuned) | 314 | 0.474 | 0.363 | +0.111 |
| **L1 learned** | 432 | **0.660** | 0.407 | **+0.253** |

- **Primary: L1 minus E0 is +0.219 in F1, 95% [+0.166, +0.271] over 72
  blocks. Shown.** L1 minus E2 is +0.186 [+0.131, +0.238].
- L1 is above chance by +0.23 to +0.28 on each of the four windows,
  2nd f027000 included, where the rules fell to chance.
- Per class (`m1_learned_per_class.json`):

  | class | precision, E0 → L1 | recall, E0 → L1 | F1, E0 → L1 |
  |---|---|---|---|
  | passes | 0.67 → 0.63 | 0.29 → 0.77 | 0.40 → 0.70 |
  | carries | 0.42 → 0.57 | 0.54 → 0.65 | 0.47 → 0.61 |

- **What this does not show:**
  - who made the action: the player is not scored;
  - whether a pass succeeded, or how far it went;
  - anything on footage of another camera or club.

  Per-player counts still need identities.

### Goals by the kick-off formation, and the shot ball gate (`shots_goals/`)

**Protocol** (`protocol.json`, `protocol_notes.json`, `gate_recheck.json`; all
hashed before use):
- **The rule:** a goal is followed by a kick-off. Before it, the people of
  both teams stand in their own halves. A formation run is a stretch of 1-s
  samples where the share of people in their own half is at least T, for at
  least N samples. A run matches a goal if it starts 15-60 s after it.
- **Choice on ground truth** (14 dev halves): T 0.85, N 2. It finds 26 of 26
  goals, with 0 false runs in 656 min; random times find about 1.
- **Re-choice on the dev half's detections** (M7 2nd, 1-fps yolov8s scan):
  the same T 0.85, N 2, 5/5 goals, 0 false runs. Frozen in
  `frozen_goals.json`.

**The registered test** (`goals_test.json`, scored once): 4 test halves, 8
restartable goals, 184 min.

| half | goals | hits | false runs |
|---|---|---|---|
| M1 1st | 2 | 2 | 0 |
| M1 2nd | 1 | 1 | 0 |
| M9 2nd | 3 | 0 | 1 |
| M5 2nd | 2 | 1 | 1 |
| **total** | **8** | **4** (Wilson [0.22, 0.79]) | **2** (0.98 per 90 min) |

- **Fails all three registered criteria:**
  - 7/8 hits needed; 4/8 found.
  - At most 1 false run allowed; 2 occurred.
  - The scoring team had to be right on every hit but one; it was right on
    2 of 4.
- Random-time runs find 0.21 goals on average (p = 0 for 4).
- **Team rule:** the frozen rule (the person nearest the spot kicks off) is
  right 2/4 times. The other rule (where play was in the 10-40 s before) is
  right 4/4, as it was 5/5 on dev. The frozen choice was a grid-order tie on
  ground truth.
- **Why it fails (descriptive, after the test):**
  - **M9 2nd:** the two detected "teams" have almost the same mean x
    (-1.2 and -2.2 m). Kit clustering did not separate them, and the share
    peaks at 0.82-0.88 after its goals.
  - **M5 2nd:** one goal had 2 samples at or above 0.85, but not
    consecutive. The false run is a 19-s stoppage.
  - **M1:** clean peaks (0.94-1.0).
- **So the rule holds where teams are told apart, and team assignment is
  the limit.** Next:
  - teams per track rather than per detection, or a split per sample
    robust to label swaps;
  - the second team rule;
  - a test on halves not yet used.

**The shot ball gate:**
- **The question:** is the ball seen near the shooter and along the shot,
  on footage v3 never trained on?
- **Data:** 26 attempts on M7 1st and M9 2nd. v3 runs on [-1, +2.2] s; the
  painted marks are dropped.
- **The criterion:** at least 2 frames with a candidate within 4 m of the
  shooter, and at least 2 frames within 4 m of the shooter-to-goal segment.
- **Result: GO, 19/26 reachable** (Wilson [0.54, 0.86]). By half: M7 1st
  3/10, M9 2nd 16/16.
- **A placebo** pairs one attempt's candidates with another attempt's
  geometry. It is reachable on 0.07 (M7) and 0.15 (M9) of pairs, so the
  hits are not clutter.
- **The reproducibility check failed** on one attempt. The planning agent's
  read-only measurement of M9 f2296 was wrong: its ball moves 5 m towards
  the goal in 0.2 s. This was recorded in `gate_recheck.json` before any
  scoring.
- **The shots test is registered** (`protocol_shots.json`): dev on the gate
  clips, then all 29 M1 attempts, recall at 2 s. Its dev failed and the
  test was not run (below).

### GPU on Kaggle (`research/session_scratchpad/kaggle/`)

- **Setup:** private Kaggle kernels driven by the Kaggle API from here, T4
  x2, no internet (wheels, weights and resnet18 come in a private dataset).
  Data go up as private datasets (CC BY 4.0). M7 / M9 clips are cut here
  through `src/unseal.py` (every opening logged); `VEO_UNSEAL` never leaves
  this machine, and the test-match guard is lifted on Kaggle only for the
  units listed in `opened.json` (owner decision).
- **Speed:** yolov8s at full width 0.10 s/frame (fp32), the v3 ball 0.12
  s/frame, against 1.0-1.6 and ~1.2 on CPU. A whole 3-minute window
  (players, teams and events_v1, dense v3) takes about 14.5 min with the
  players and the ball on separate GPUs, about 2.5 h on CPU. A 57-s clip
  still costs ~10 min (fixed costs per run).
- **GPU equals CPU:** on 7 windows the player rows, tracks, teams and
  events_v1 are identical to the CPU runs (one extra player row in
  ~40,000 on M6 1st); the v3 rows are identical where a CPU run exists (max
  shift 0.0008 px). `compare_*.json`.
- **Backend note** (`backend_note.json` + addendum, hashed before the
  shots dev): all on T4 fp32, supervision 0.30.9, ultralytics 8.4.183.

### Shots (`protocol_shots.json`): the dev fails, the test is not run

- **Dev:** 8 windows and the 26 gate attempts as clips (-15 s / +10 s,
  cut with 400 frames each side for v3's background; owner decision), all
  on the T4. Two M7 attempts lie both in a clip and in the M7 window: they
  count once, in their clip.
- **Frozen** (`frozen_shots.json`): 40 km/h, 35 m, mouth margin 3 m.
  **Dev F1 0.053: 1 of 32 attempts found, 5 false shots.** No cell of the
  80 finds more than 1.
- **Why** (`shots_dev_diagnostic.json`, after the freeze): the data are
  sound (both teams and an attacked goal on every clip, the ball on 25-60%
  of detected frames). The rule is too strict for the linked v3 ball, and
  the attacked-goal filter removes about half of what remains. Off the
  grid, recall rises only with 15-25 times more events than shots (21/32
  with 506 events).
- **The M1 test was not run** (owner decision, `shots_decision.json`): it
  needs 16/29, and dev reaches 1/32. This is a deviation from the
  protocol; no M1 shot output exists.
- **So shots are not available from this rule.** A later attempt needs a
  different detector (for example the learned spotter extended to shots),
  a team filter that does not rest on kit clustering, and a new
  registration.

### Learned shot spotter, tested on M5 (`shots_learn/`)

- **Protocol** (`protocol.json`, hashed before any data): the learned
  pass/carry spotter's features plus goal geometry, one class 'shot';
  trained on the clips around all 210 SHOT/GOAL of M1, M2, M3, M4, M6, M8,
  M10 (194 clips) plus 7 windows; test M5 (29 clips, 33 attempts), batches
  in a registered random order with an early stop when success is out of
  reach (owner decision). Success: recall >= 18/33 at 2 s and precision >=
  0.50. All on the Kaggle T4 (clips cut here, ~1.2 min each, uploaded).
- **Dev** (nested leave-one-match-out, 216 shots): recall 0.54, precision
  0.52; gate (0.45 / 0.40) passed; frozen (threshold 0.825). Weak on M1
  (7/29) and M6 (8/30), strong on M2 (24/29) and M8 (22/29).
- **Test: fails by one shot.** Recall **17/33 = 0.52** (Wilson [0.35,
  0.68]); precision **0.61** (17 of 28). The early stop never fired. The
  rule on the same spans: 2/33.
- **Disclosed:** v3 was trained on M5 (its provenance), so the ball ran
  with --allow-seen; M5's shot frames were read to plan the clips.
- **Reading:** the learned spotter finds about half the shots at about 60%
  precision, far above the rule, but it is not shown to reach the
  registered bar. Next: more training shots (M7 / M9 are spent), the
  shooter's identity, and a stronger ball near the goals.

### Goals, revisited (`goals_v2/`, `teams/`; exploratory, no GPU)

- **The teams are not the cause of M9's misses.** The 1-fps scan of M9 2nd
  (re-run here) has its teams right on 0.977 of matched outfield players;
  per-detection kit clustering is accurate pooled over the half (0.98).
  The earlier reading ("teams not separated") came from the half-mean x,
  which is close for both teams in a one-sided half (-9.0 / -3.7 m in the
  ground truth).
- **The registered rule is right on the ground truth**: 41/41 restartable
  goals of the 20 halves, 0 false runs (M9's shares 0.91-0.96). The test's
  misses come from the detections' share (0.82-0.88 on M9: team errors
  and positions across the halfway line, 13-17 of 22 people seen).
- **A kick-off cue (a detection within 3 m of the centre spot) with T
  0.80**, chosen on the scans of M5 2nd and M9 2nd (5/5, 0 false there,
  against 1/5, 2 false), was frozen before the M1 scans ended and then
  **failed on M1: 0/3**, where the registered rule gets 3/3. On the ground
  truth it is also worse (37/41, 4 false). Not adopted.
- **So the goals rule stays as registered**; what would help is a cleaner
  detection share (fewer team errors near the halfway line, more players
  seen), not a new rule.
- **Per-detection teams beat per-track teams** on 7 windows (smoothed over
  +-2 s of the same track: 0.92-0.98 against 0.79-0.93): a candidate for
  the main pipeline (needs the GPU to recompute kit features on clips).

### Who made the pass (`passer/`; exploratory, CPU only)

The actor of each labelled BAS action, from detections only, scored apart
from event detection (`plan.json`, hashed before scoring; `addendum1.json`:
two fixes after the first runs, rules unchanged). Rule A: at the label
frame, the v3s_link ball within 0.2 s and the player detection nearest it
in metres, if within 3 m. Rule B: the best (player, frame) over -0.4/+0.2 s.
Truth: the detection's ground-truth person (feet; M1 in pitch metres under
the refit) against the label's player_id.

| | n | right, end to end (A) | precision (A) | coverage (A) | ceiling |
|---|---|---|---|---|---|
| passes, matches v3 was trained on | 1847 | 0.253 [0.234, 0.273] | 0.724 | 0.349 | 0.806 |
| all actions, same matches | 3599 | 0.257 | 0.717 | 0.359 | 0.800 |
| passes, M1 (unseen) | 214 | 0.061 [0.036, 0.101] | 0.186 | 0.327 | 0.280 |

- **Correction (2026-10-26, `ball_range.json`):** the clips' ball was
  detected only on local frames 350-1075 (the per-clip cost cut); half of
  the labelled actions sit in the margins, where no ball exists. Inside
  the ball range rule A is right on **0.487** of all actions (passes
  0.502), precision 0.717, coverage 0.679; the table above counts the
  margins.
- **The ball is the bottleneck.** On the seen matches the linked ball is
  there within 0.2 s of only 47% of the actions; given a ball, a player is
  within 3 m 77% of the time and is the right one 72% (82% when the
  labelled player is detected at all). Rule B adds 3 points.
- **M1 is not measurable here:** the labelled player is matched to a
  detection on 28% of its actions (80% elsewhere), so attribution and
  evaluation errors cannot be told apart (M1's camera and truth boxes are
  off). M7 / M9 have no linked ball track for their clips: not scored.
- The seen-match figures are optimistic (v3 trained on those matches).

### Inferring the ball in gaps (`ball_interp/`; quick feasibility, CPU)

Two M6 windows, 320 hand-verified balls (v3 trained on M6). The link
(`src/ball_link.py`, a Viterbi that already skips up to 5 frames) covers
0.95 of them at precision 0.99. Linear filling of its gaps:

- **Real frames:** coverage 1.00, precision 0.96-0.97 (12 px).
- **Simulated occlusions** (detections removed around verified balls),
  within 30 px: 0.5 s holes 0.70, 1 s 0.49, 2 s 0.18. Following the
  player's feet instead of the line is worse (0.57 at 0.5 s).
- **Who passed:** no gain (rule A 0.253 -> 0.249 / 0.250 with 0.5 / 1 s
  fills): the gaps around actions are mostly longer than 1 s.
- So filling works for short gaps only. The "2-3x more frames" of a
  direct link is explained (`passer/ball_range.json`): the pipeline links
  on the players' frames only (every 2nd frame) and cuts candidates below
  p 0.10 first (about 1.25x); no later stage loses balls. Rerun inside the
  ball range (`passer/fill_check_range.json`, 936 passes): still no gain.
  Rule A 0.500 -> 0.491 / 0.494 with 0.5 / 1 s fills; rule B 0.558 ->
  0.552 / 0.556 (coverage 0.81 -> 0.85, precision 0.69 -> 0.65: the added
  actors are mostly wrong). Inside the range the limits are precision
  (28% wrong when someone is named) and the ceiling (23% of labelled
  players never matched to a detection), not ball gaps.
- **Why rule A names the wrong player** (`passer/diagnose.json`; 367 wrong
  of 1293 named, in range, seen matches): an opponent 61%, a teammate
  22%, a detection with no truth person 17%. The labelled player is not
  detected at that frame in 44% of them. When it is, it is a median 3.3 m
  from the ball against 1.45 m for the named one, within 1 m of it in
  41% (contests). In 65% the labelled player is the nearest at some
  instant within +-1 s, most often 0.2-0.4 s before the label (the label
  sits slightly after the touch).
- **Rule C, possession** (`passer/possession.json`; the track most often
  nearest the ball over a window, votes at every players' frame): on the
  same 1905 actions, A 0.486, B 0.540, C [-0.4, +0.2] s **0.568** [0.546,
  0.590] (passes 0.586), precision 0.681, coverage 0.834. Paired with B:
  90 actions gained, 65 lost (sign test p ~0.05). Longer windows cover
  more but lose precision. The window is chosen in-sample on matches v3
  was trained on: a small gain, not shown on a held-out match.


### Distance and possession (`events_stats/`; descriptive, weak evidence)

**The report's estimators, as they are** (`stats_eval.py`; M6 2nd, M7, M9;
v3 ball, static filtered, linked):
- Distance: ×2 the truth in total (7.7 / 6.5 / 7.0 km against
  3.6 / 3.6 / 3.4). Position jitter at 25 fps adds up.
- Possession (`possession_proxy`): the left side's share is wrong by
  0.23-0.27.

**Retuned on those three windows** (`stats_fix.py`, `stats_fix_tune.json`):
- **Distance:** a centred moving average of 1 s, then a sample every 1 s.
  The per-person median error falls from 0.32 to 0.046.
  - **The comparison is circular.** The truth's reference is the same
    1 s / 1 s estimator, because its positions sit on a 1.05 × 0.68 m grid
    and jitter too. Path length depends on the sampling scale.
  - What this shows is that our positions give the same path length as the
    truth's at a 1 s scale. It does not show that this is the distance run,
    which a 1 s path underestimates on turns.
  - It is measured with a perfect identity, on the frames where a
    detection was matched to the person.
- **Possession:** the team of the nearest player within 4 m of the ball, at
  every frame with a ball, with no ball-speed gate. The error falls to
  0.086, in-sample. A constant 0.5 scores 0.148 there.

**On M1 under the refit**, with detections matched to persons in pitch
metres within 3 m (`--pitch`; M1's truth boxes are in another frame):

| window | distance error (1 s / 1 s) | distance error (every frame) | possession error, retuned | possession error, current proxy | possession error, constant 0.5 |
|---|---|---|---|---|---|
| 1st f001500 (*) | 0.035 | 0.62 | 0.008 | 0.078 | 0.100 |
| 1st f024000 | 0.020 | 0.56 | 0.084 | 0.050 | 0.015 |
| 2nd f000000 (*) | 0.033 | 0.80 | 0.028 | 0.016 | 0.143 |
| 2nd f027000 | 0.035 | 0.89 | 0.299 | 0.435 | 0.052 |

(*) used by the refit, so in-sample for the positions.

- **Distance:** the 1 s scale holds on M1, with the caveats above. The
  matched frames cover 0.52-0.74 of the truth's distance.
- **Per-player km over all frames, with our identities**, is the product's
  number. It is not measured yet, and identities are the limit.
- **Possession: not shown better than a constant 0.5.** The mean errors
  are 0.105 (retuned), 0.145 (current) and 0.078 (constant 0.5).
- The 2nd f027000 window is far off for both rules, and is also where E2
  fell to chance. The cause is not yet found; team assignment is the first
  suspect.
- A fair possession test needs finer units (30 s bins, or frame-level
  agreement with dead time excluded), a paired comparison with a constant,
  and tuning leave-one-match-out.

## Next

0. **Balls at rest by the goals** (seen on M9, judged B*). Find them
   from the window's own candidates, like the painted spots, and drop
   them before tracking. This has to be tested on footage not yet seen,
   or on clicked balls; M7 / M9 are spent.

1. **The ball in flight.** In this attempt (one seed, 63 tuning passes),
   weak labels from the actions did not beat v1.
   - **What was tried (v2):** pass corridors, 5,000 ball-free strips, both
     halves of four matches, and motion input.
   - **The result:** v2 equals v1 by FF-auto on M6 (0.63 against 0.62;
     0.64 against 0.62 on its second half, unseen by both). v1 leads on
     the legacy measures, and the labels alone scored below v1.

   Next:
   - **Clicked balls** (`make_ball_labeller.py`): a few hundred in-flight
     frames per match. They would be the first labels that are the ball
     itself, and also a test set that needs no corridor rule.
   - **Tracking across frames:** tried in protocol v4 (`src/ball_track.py`,
     stride 2, gaps of up to 6 frames filled on the pitch). It is not
     shown better than v2's raw pick on V-new (18 against 9 passes, p
     0.061; underpowered). At detection frames it was never worse. The
     next version should track at stride 1, and be tested on M7 / M9 or
     clicked balls.
   - **Higher input resolution on the far side,** where the ball is 5 px.
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
