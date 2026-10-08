# Who did it: player identity, measured

Every event the pipeline emits names a player, and until now nothing had
checked it. This document records what was measured, in order, and what
it means for crediting events to players.

## The plan

Asked for: anonymous identities -- one per person, the same all match --
then a shirt number read wherever it is legible and carried to all of that
person's frames, then the squad list to put names on numbers.

## Ground truth

- **Clicks on our footage.** `make_player_labeller.py` showed 139 events
  sampled across the six labelled windows (passes, carries, recoveries, a
  tackle, every labelled shot); the person clicked the player who played
  the ball and typed the number where readable. `score_player_labels.py`
  scores against it.
- **SoccerNet game-state clips** (SN-GSR-2025, public): 30-second
  broadcast clips at 1080p where every person keeps one id for the whole
  clip, with team, role and number. Ten validation clips were fetched
  (byte ranges of the archive: 9 s a clip instead of hours) and run
  through the full pipeline; `eval_identity_gsr.py` matches our
  detections to theirs by overlap and scores identities: people split
  (identities per person), purity, IDF1.

## What the clicks say about events

Of the 139 events, the person marked as not happening: carries 29 of 46,
recoveries 27 of 34, passes 16 of 46, the one tackle. On the 66 real
ones, the pipeline credited the clicked player 22 times. Every clicked
player was detected; the faults are upstream:

- passes are stamped a median 10 frames (0.4 s) after the touch, when the
  ball is already in flight, so "nearest player at the event frame" is
  often someone else; crediting the player at the touch instead gives 29
  of 66;
- in 21 of 66 the tracked ball never reaches the clicked player;
- the person's notes: one pass counted twice, a team-mate's reception
  called a recovery, a carry in the middle of a pass, pass timing that
  wanders. All four are what a detector built on "nearest player over
  time" rather than on touches does.

Only 10 numbers could be read by eye in 139 events.

## Shirt numbers

`train_jersey_reader.py` trains a reader (ResNet-18: number visible,
tens, units) on the SoccerNet jersey set, crops shrunk to this footage's
sizes. On 142 held-out tracklets of that set it reads 91 of 101 numbered
ones right; tuned (a crop counts at 0.95 visible, a track needs 60% of 5
crops), 82 right, 4 wrong, and 27 of 41 that show no number left unread.

On broadcast it does not transfer. On game-state clip SNGS-021, from the
ground truth's own tight boxes, it named none of the 9 people with a
legible number right (2 wrong, 7 unread); only 9 of the 21 people there
have a legible number anywhere in the 30 s. Numbers therefore no longer
join tracks by default (`identify_players.py --numbers` turns them on).

## Anonymous identities

`src/player_identity.py` joins a clip's tracks into people. Two tracks on
screen together are two people; within a team, tracks are joined by an
appearance embedding (`train_player_reid.py`) trained on 15 game-state
training clips with batches drawn from one clip, so it learns to tell
team-mates in the same kit apart. On held-out clips it ranks the same
person first among team-mates 93.4% of the time (mAP 90.3%), against
42.5% (23.0%) for ImageNet features.

A pair model (`train_identity_pairs.py`) adds where and when tracks are --
frames shared, the gap, the jump from where one ended to where the next
starts, size -- because appearance alone overlaps: the same person's
tracks are 0.71-0.98 alike, team-mates 0.68 at the median and 0.91 at the
90th percentile.

On the ten game-state clips, each scored by a pair model fitted on the
other five:

| | identities (21-23 people) | split per person | purity | IDF1 |
|---|---|---|---|---|
| re-joined tracks | 46.1 | 3.33 | 0.873 | 0.617 |
| by appearance, 0.9 | 41.0 | 3.13 | 0.862 | **0.645** |
| pair model, 0.5 | 37.3 | 3.02 | 0.846 | **0.646** |

A small gain, the same either way; appearance alone is the simpler and is
the default. The ceiling is set before joining: the tracker's own tracks
are 87% pure, so they already switch between people when players cross,
and no joining undoes that. The next gain is the embedding inside the
tracker, so tracks stop switching in the first place.

On the six labelled broadcast windows, 96-138 tracks become 58-81
identities. Of the clicked players with different typed numbers none were
wrongly joined; the three pairs with the same number stay split.

## Where this leaves crediting events

Identity is not what limits it today: the wrong credits go to other
people, not to another fragment of the right one. The event detector is
the limit -- timing, events that did not happen, and the ball track.
`src/event_schema.py` sets the format events should come out in (SPADL,
with start and end frames and both players), which is also the format a
next labelling round should use.

## Events from touches

`src/touch_events.py` rebuilds events on contact instead of "nearest
player over time". A player has the ball from a touch (the ball within 0.6
body heights of their feet) and keeps it while it stays within 0.9 and
nobody else touches it; a possession must last 2 frames. Actions follow
from consecutive possessions, in the SPADL format: a pass from the
release to a team-mate's reception (the ball travelling at least a body
height), a failed pass and a recovery when an opponent receives it, a
tackle when an opponent takes it at close quarters and keeps it 4 frames,
a carry when a player keeps it 0.6 s and moves a metre.
`detect_touch_events.py` writes them as a table.

Against the 139 clicks (the settings were chosen on them, so this is
optimistic):

| | found | right player | rejected moments still detected |
|---|---|---|---|
| passes (30 real) | 19 | 14 | 5 of 16 |
| carries (17 real) | 9 | 7 | 5 of 29 |
| recoveries (7 real) | 4 | 3 | 6 of 27 |

The old detector made all 72 of the rejected events; the new one makes 16
of them. Its rates are plausible -- 18 passes, 7 carries, 5 recoveries a
minute, no pass counted twice -- except tackles, 5-7 a minute, which are
likely duels and are not measured (one tackle was clicked, and it did not
happen). On "right player" the comparison is not fair either way: the
clicks sampled the old detector's events, so it "found" all of them and
credited the right player on 27 of 54; the new one has to find them first
(32 of 54) and credits the right one on 24.

A fair measure needs actions labelled independently of any detector:
every action in a stretch of play, with start and end frames and both
players, in the format of `src/event_schema.py`.

### Measured on every action in six stretches of play

`make_action_labeller.py` asked for every action in six 10-second
stretches of live play, chosen by ball coverage rather than by any
detector: 32 actions (21 passes, 5 carries, 6 balls won). Neither
detector's settings were chosen on these labels. `score_action_labels.py`,
matching one to one within 0.4 s:

| | labelled | old: found / detected / right player | new: found / detected / right player |
|---|---|---|---|
| passes | 21 | 7 / 17 / 6 (receiver 2 of 5) | **16 / 20 / 14 (receiver 8 of 12)** |
| carries | 5 | 3 / 23 / 2 | 4 / 12 / 3 |
| balls won | 6 | 0 / 3 / 0 | 3 / 11 / 1 |

Passes: recall 33% -> 76%, precision 41% -> 80%, the end of a pass within
a median 4 frames of the click against 18. Carries and balls won are
still over-produced -- 12 carries and 11 tackles or recoveries detected
for 5 and 6 labelled -- most likely possessions held at a player's feet
without moving, and duels read as tackles.

### Fewer carries and balls won: a modest trade

Tuned on the first round of clicks only, keeping the stretch labels as a
held-out test: a carry now moves the player at least a body height in the
camera-compensated picture; a tackle needs the winner to keep the ball 8
frames, a recovery 4. On the clicks: false carries 5 -> 3 of 29 rejected
moments, tackles 5.7 -> 2.9 a minute, with one real carry lost.

On the held-out stretches, once:

| | labelled | before: found / detected | after: found / detected |
|---|---|---|---|
| carries | 5 | 4 / 12 | 3 / 8 |
| balls won | 6 | 3 / 11 | 2 / 7 |
| passes | 21 | 16 / 20 | 16 / 20 |

A third fewer false carries and balls won, one real one of each lost, so
precision barely moves (carries 33% -> 38%, balls won 27% -> 29%). With
5 and 6 labelled these are within noise. What the remaining false ones
are needs looking at on the footage before another rule is added.

What the remaining false balls won are, looked at on the footage (from
the first round of clicks, to keep the stretches held out): one is on a
cut to a close-up camera, one has the tracked ball up by the hoardings,
and two are next to the referee. Those two were first read as the
referee winning the ball; checked afterwards, the track credited at both
is an outfield player of one team (the role classifier gives it 100% and
90% player) and the referee stands about a body height away, so they are
contested balls near the referee, not the referee holding it. Excluding
unplaced people ("other") and skipping close-up frames were both measured
and left off.

### Referees: recognised on SoccerNet, not yet on these windows

`train_role_classifier.py` learns player / goalkeeper / referee from the
crop on the 15 game-state training clips (balanced classes). On the ten
validation clips -- matches it never saw -- with the ground truth's boxes
it finds 20 of 20 referees and calls no outfield player a referee (5 of 11
goalkeepers are called referees); through our own tracks, 95% of referee
detections are called referee and 0.7% of players' detections.

On the six 720p broadcast windows it does not transfer: of the 66
clicked players it calls 2 referees and 23 goalkeepers -- some of the
clicks are goalkeepers' passes, so not all 23 are wrong, but there are
only two keepers a match -- and on the Reading windows about 45% of
player detections "goalkeeper". Keeping
"referees" off the ball (`touch_events.EXCLUDE_REFEREES`) removed one false
recovery on the clicks and lost 2 real passes there and 1 on the held-out
stretches, so it is off. The gap is the footage -- lower resolution and
these kits -- and closing it needs referee examples from it. It would
not have fixed the two false tackles near the referee either: the track
credited there is a player (see above).

### A second round of stretches

Six more 10-second stretches, not overlapping the first (`make_action_labeller.py
--exclude ... --key action-label-v2`), labelled the same way, with a key
for the goalkeeper: 32 actions (23 passes, 3 carries, 3 balls won, a
shot). No setting was chosen on them, so they are a clean test.

| passes | round 2: old -> new | both rounds (44 passes): old -> new |
|---|---|---|
| found | 26% -> **70%** | 30% -> **73%** |
| detections real | 35% -> **67%** | 38% -> **73%** |
| right passer | 2 of 6 -> **12 of 16** | 8 of 13 -> **26 of 32** |
| right receiver | 1 of 5 -> **6 of 13** | 3 of 10 -> **14 of 25** |
| end of the pass, median | 20 -> 4 frames | 20 -> 4 frames |

Over both rounds, carries: 6 of 8 found, 16 detected; balls won: 2 of 9
found, 15 detected; the one shot is not in this detector. Balls won and
pass receivers are the weak points now.

The goalkeeper marked in stoke_7001 -- who received one pass and played
the next -- is called a player with certainty by the role classifier;
team assignment had put him in "other".

### Receivers and balls won

What went wrong on round 1 (used to tune; round 2 kept for testing): a
failed pass had no receiver though the person clicked the opponent who
got it; a player the team assignment could not place was taken as a
team-mate; a ball brushing a player in flight ended the pass there; and
recoveries were dropped by the 4-frame hold. The receiver of a failed
pass is now the opponent who got it, and a recovery needs only the touch
(1 frame); treating unplaced players as opponents, and a 4-frame hold for
a reception, both measured worse on round 1 and are not used.

| | round 1 (tuned on): before -> after | round 2 (test): before -> after |
|---|---|---|
| right receiver | 8 of 12 -> 9 of 12 | **6 of 13 -> 8 of 13** |
| balls won found / detected | 2 / 7 -> 3 / 8 | 0 / 8 -> 1 / 11 |
| passes found / detected | 16 / 20, unchanged | 16 / 24, unchanged |

Receivers improve on the test. Balls won are within noise on 3 labelled;
over both rounds 4 of 9 are found with 19 detected, against 2 of 9 with
15 -- still the weakest action.

## Tracks that jump between players, cut and rejoined

The tracker's tracks are 87% pure: when players cross, a track carries on
with the wrong one, and no joining afterwards undoes it.
`src/track_split.py` samples the appearance embedding every 5 frames along
each track and cuts where the look of the 3 samples before a point and the
3 after are less than 0.4 alike; the pieces are then rejoined by look at
0.7 (`identify_players.py --split`, which writes `tracks_split.parquet`;
the event detector and the scorers use it where it exists).

Chosen on five game-state clips (`tune_track_split.py`), on the other five:

| | identities (21-23 people) | purity | IDF1 |
|---|---|---|---|
| tracks as they are | 44.6 | 0.889 | 0.638 |
| joined by look (before) | 40.2 | 0.878 | 0.656 |
| **cut, then joined** | **36.2** | 0.863 | **0.683** |

On the six labelled broadcast windows the 112-190 pieces become 33-54
identities for about 25 people (58-81 before). The credit for each event
does not change -- an event goes to the track at the ball on that frame,
which splitting does not move -- and the three pairs of clicks with the
same typed number are still two identities each; what improves is one
person keeping one name across the clip, which is what per-player
statistics add up.

## One name across the match

`src/match_identity.py` links each clip's identities across all the clips
of a match by look alone -- clips are minutes apart, so nothing else
connects them. Teams are formed from the looks across the whole match
(two clusters), unplaced identities included; within each team,
average-link clustering at 0.8, never joining two identities of one clip
that are on screen together. `match_players.py` writes
`match_identities.json`; the event detector and the scorers then name
players match-wide.

Scored on ten game-state clips of one game, where a person is followed
across clips by side and shirt number (people with no legible number are
left out), over pairs of identities from different clips:

| | five clips (used to choose) | five other clips |
|---|---|---|
| each clip's team labels lined up | 75% right, 48% found | 56% right, 17% found |
| unplaced identities to the nearest team | 75% / 48% | 63% / 24% |
| **teams from the looks across the match** | 69% / 44% | **69% / 34%** |

The first row is best on the clips it was chosen on and falls apart on
the others: half the true pairs had been put in different teams, 20
identities in "other" alone. Teams from the looks hold up, and are used.
What it gives is partial: about 7 in 10 links right, and a third to a
half of a person's identities linked.

On the labelled windows, 141 identities become 83 players for Stoke v
Huddersfield and 123 become 67 for Reading v Fulham, for about 28 people
each -- each person still in about three pieces. Of the typed numbers,
no two different numbers in different clips were given one name; no
number was typed in two clips, so the links themselves cannot be checked
there, and three clicks on #9 in one clip still carry three names.
Event credit is unchanged.

To check the links on our own footage, `make_number_labeller.py` builds a
page that plays the clips (the person opens their own copies; nothing is
embedded) and lets a person pause wherever a shirt number is legible --
close-ups, mostly, since on the main camera players are 60-150 px tall at
720p -- click the player's feet and give the team and number. A first
version showed still frames chosen by the pipeline and could not be used:
main-camera frames showed no legible number. `score_player_names.py`
then counts, within a clip and across a match, same-number pairs joined
and different-number pairs kept apart, and how often the team is right.
A first look already shows close-up tracks often left without a team
(382 of 1015 detections taller than 180 px in one clip are "other").
