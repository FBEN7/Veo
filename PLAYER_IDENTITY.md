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
