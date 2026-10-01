# Event detection against SoccerNet ball actions

The only event ground truth this project had was twelve hand-annotated
intervals, which is too few to tell a working detector from a lucky one.
SoccerNet's ball action spotting set timestamps every Pass, Drive, Header,
Cross and Shot, so a 90-second window carries 60 of them.

Clip: 2019-10-01 Stoke City vs Huddersfield Town, 720p, match time 1820-1910 s
(the densest 90 seconds in the half). Labels in window: 34 PASS, 25 DRIVE,
1 HIGH PASS.

Reproduce with:

    python score_soccernet.py --clip soccernet_clip.mp4 \
        --labels Labels-ball.json --offset 1820 --out output_soccernet

SoccerNet data is under a non-commercial NDA. It is a measuring instrument --
nothing derived from it belongs in the product or in this repository.

## Current state -- read this first

This document is chronological: findings are appended as they are made, and
later sections correct earlier ones rather than rewriting them. The state as
of the latest measurement:

| | result | section |
|---|---|---|
| Camera position | located on 6 of 6 clips; three Reading clips agree to 1.1 m | "Where the camera is" |
| Goal-box poses | 0.53 m median from corner poses (was 18 m) | "What a goal box is" |
| Ball placed on the pitch | 8-45% of frames by clip | "What it changes" |
| Shot distance, angle, xG | all 6 labelled shots, total xG 0.47 | "Where the camera is" |
| Out of play | precision 2 of 2, recall 2 of 4 | "The continuity check", "The camera between grid frames" |
| Shots and goals detected | **3 of 10, 1 of 2**, no false ones | "Shots at the goal plane" |
| Ball height | works on simulated 1 s flights; not yet on these | "Ball height" |

Superseded, and kept only as history: anything placed by the centre-circle
**anchor** (quarantined: its camera sits 62-72 m up), the **1.8 m** box-pose
error (it compared two poses sharing a wrong camera), the **"1 of 8" shots**
(a cross) and **"1 of 2" goals** (crowd detections), and the out-of-play
reasoning that the margin was smaller than the placement error.

The limits now: four labelled shot moments have almost no placed ball
positions, and outside the goal plane an airborne ball is still placed as
if on the grass.

## Pass outcomes are close to a coin flip

The model reports 50%, 61%, 51% and 55% of passes as intercepted across the
four windows. SoccerNet's own team labels give the truth: consecutive ball
actions change team only 3%, 23%, 16% and 7% of the time, about 12% on
average. Turnovers are over-reported four to five fold.

This matters differently from the timing results. Pass detection being above
chance means the *moments* are roughly right; this says the *labels attached
to them* carry little information. A coach reading "19 completed, 19
intercepted" is reading fiction, and unlike a timing error it is obvious to
anyone who knows football.

One candidate cause is ruled out:

* **It is not fragmentation.** Zero intercepted passes involve a passer and
  receiver whose tracks never coexist, so they are genuinely distinct players,
  6 to 11 m apart.

A second was reported as ruled out and was not. "Team does not flicker within
a track -- zero of 646 tracks ever change team" is **true by construction and
proves nothing**: `team_assignment.py` ends with
`tracks["team"] = tracks.track_id.map(team_map)`, one team per track id, so a
track cannot change team whatever the video shows. The test could only ever
return zero. It also gave false confidence, because a track *can* change
player -- see `TEAM_ASSIGNMENT.md`, where one track visibly switches from a
red-striped player to a navy one mid-track, which a single per-track label has
no way to express.

Team assignment error is a contributor but provably not the main one.
If each end of a pass is assigned correctly with probability a, the reported
turnover rate is

    0.88 * 2a(1-a)  +  0.12 * (a^2 + (1-a)^2)

which peaks at 50% when a = 0.5. Three of the four windows exceed that, so
something beyond team assignment is inflating the rate.

The requirement this sets is worth stating plainly: for the reported rate to
fall below 20% against a true 12%, per-end team accuracy must reach about
0.95. It currently measures 0.73 to 0.81. That is a different order of
problem from a threshold.

### The outcome field is independent of the outcome

Comparing rates is a weak test: two rates can agree while every individual
call is wrong. The labels support the strong test as well, because the team
of the *next* ball action says whether a pass kept possession. Scoring our
call against that truth, pooled over the four windows
(`python analyse_pass_outcome.py`):

|  | we said success | we said intercepted |
|---|---|---|
| pass kept the ball | 34 | 44 |
| pass lost the ball | 7 | 9 |

We call "intercepted" on **56%** of the passes that kept the ball and on
**56%** of the passes that lost it. Fisher exact p = 1.00. The field is not
merely inaccurate; it is statistically independent of the thing it names.

Restricting to the passes whose passer's team we did attribute correctly
barely moves it -- 46% against 62%, p = 0.37, on 13 genuinely lost passes.
Per-window outcome accuracy rises from 45% to 54% under that restriction,
against a baseline of 82%. **Perfect team assignment would not fix this.**
That rules out the hypothesis the rate comparison above suggested, and moves
the fault into how possession transfer itself is decided.

The remaining contribution is not isolated. Intercepted passes coincide with
two to three times higher ball speed on three windows (112 against 35 km/h on
w1) and the reverse on the fourth, and the test is compromised because
`_ball_kinematics` clips speed at 120 km/h, hiding exactly the spikes measured
elsewhere. Suggestive, not shown.

### Measured on the terms that matter to a user

Timing precision is not what this field is judged on, so scored at +/-2 s
where a second of error costs nothing:

| | recall | precision |
|---|---|---|
| pass | 0.89 / 1.00 / 0.93 / 0.71 | 0.82 / 0.52 / 0.64 / 0.69 |
| carry | 0.96 / 0.95 / 1.00 / 0.88 | 0.57 / 0.53 / 0.56 / 0.45 |

Recall is good: almost everything is found. Precision is the fault, at roughly
two events emitted per event that exists.

Outcome accuracy on the passes that match a real pass is **52%, 55%, 43% and
30%** (`analyse_pass_outcome.py`, matching passes at +/-2 s). The trivial
baseline of always answering "success" would score **94%, 65%, 78% and 90%**
on the same passes. The field is about 37 points worse than a constant: it
carries negative information.

### Six hypotheses eliminated, cause not found

* ~~**Team flicker within a track**~~ -- withdrawn. The test was vacuous: team
  is assigned once per track id, so zero was the only possible answer.
* **Fragmentation** -- zero intercepted passes involve a passer and receiver
  whose tracks never coexist; they are distinct players 6-11 m apart.
* **Thin evidence on short tracks** -- essentially every matched event comes
  from a track of 100+ frames (median 190-220), where accuracy is still
  0.71-0.83.
* ~~**How the colour is measured**~~ -- withdrawn. The rebuild was scored at
  0.775 against 0.778 on the compound metric. Against hand-read kit labels it
  is 0.98-0.99 against 0.93-0.97, so it was better all along and the metric
  hid it. See `TEAM_ASSIGNMENT.md`.
* **Ball position unreliability at the event** -- team accuracy is 0.79 when
  the ball is slow at the event and 0.78 when fast, pooled over 175 events.

A sixth is now eliminated too: **team assignment is not the binding
constraint**. Conditioning on a correctly attributed passer leaves the call
independent of the truth (above). Fixing the kit clustering would raise
outcome accuracy from 45% to at most about 54%, still 28 points below
answering "success" every time.

What remains untested is whether the possessor is the right player at all.
Possession is nearest-player, and a nearby opponent would produce exactly this
signature. SoccerNet's ball-action file has five fields -- gameTime, label,
position, team, visibility -- and no player identity, so it can say whether
the *team* is right (it does, above) but not whether the *player* is. Note
that the player question only reaches the outcome through the team: picking
the wrong player on the right team leaves the outcome unchanged. Player-level
ground truth exists in SoccerNet's Game State Reconstruction set, which is a
separate download from the ball-action spotting set used here.

### The rebuild: deciding the receiver once the ball settles

The fault was in *when* the receiver was read. Possession changed hands the
instant the ball came near somebody, so the receiver was whoever the ball
travelled past -- on the wrong calls, an opponent 2.1 m away with the nearest
teammate at 3.7 m. The right calls are the mirror image (teammate 2.8 m,
opponent 4.3 m), so the two are symmetric and no threshold on that separation
separates them. `diagnose_transfer.py` measures this; the receiver is the
nearest player 100% of the time.

`src/events.py` had a `_settled_possessor` whose docstring described the fix
exactly -- read the receiver once the ball is under control -- and which was
never called from anywhere. It is now `_settled_receiver`, wired in, with
three gates: read the geometry at the first controlled frame within
`RECEIVER_SETTLE_WINDOW_S`; require the player to be within
`RECEIVER_MAX_DIST_M`; require them to be clear of the nearest opponent by
`RECEIVER_TEAM_MARGIN_M`. Failing a gate emits the pass with outcome
`"unknown"` rather than a guess.

**A trap worth naming.** The truth here is the team of the *next* labelled
ball action, and the next action follows a pass after a median of 1.08-1.40 s,
with 52-69% inside 1.5 s. A rule that waits 1.5 s and reads off who has the
ball is therefore reading the answer. It scored balanced accuracy **0.89 on a
held-out match**, better than anything defensible, and it means nothing. The
settle window is capped at 0.6 s for this reason, not because 0.6 s measured
best. At that cap only 0-5% of next actions have occurred.

Swept over settle, radius and margin, fitted on the three Stoke windows and
checked on the unseen Reading match (`experiment_receiver.py`). Balanced
accuracy is the measure -- 17% of these passes lost the ball, so a constant
"success" scores 83% accuracy and 0.50 balanced:

| | tuning windows | held-out match |
|---|---|---|
| current behaviour | 0.59 | — |
| settle 0.3 s, 6 m, 1 m margin | **0.80** | **0.83** |

That is with the passer's team taken from SoccerNet. With our own team
assignment the same rule gives 0.71 on the tuning windows and **0.28** held
out -- worse than chance. The held-out window is the one where team
attribution measures 0.55, which is a coin flip, so both ends of the pass are
noise there.

**Both ends have to be right, and that reconciles the earlier result.**
Conditioning on a correct passer did not help while the receiver logic was
broken; fixing the receiver does not help where the passer's team is random.
Neither fix shows on its own.

End to end, pooled over the four windows, the rebuild moves:

| | before | after |
|---|---|---|
| turnovers called (truth ~17%) | 56% | 43% |
| outcome accuracy | 45% | 56% |
| ... on correct-passer passes | 54% | 70% |
| on kept vs lost, correct passer | 46% vs 62%, p=0.37 | **31% vs 62%, p=0.13** |

The separation is now 31 points in the right direction where it was 16, and
pooled over everything the call is no longer identical on the two classes
(43% vs 50%, p=0.74, against 56% vs 56%, p=1.00). None of this is
significant: there are 10 genuinely lost passes among the decided ones, which
is too few to reach p<0.05 at any effect size worth having. The mechanism is
established; the end-to-end result is not.

Detection is unaffected, which was the constraint: pass stays above chance on
4/4 windows at +/-0.5 s and +/-1 s (p 0.000-0.039), and precision/recall at
+/-2 s move only on w3, 0.64 to 0.62.

About 62% of matched passes now get a verdict; the rest say `"unknown"`.
Downstream counts completed passes as `outcome == "success"` and turnovers as
`"intercepted"`, so an unknown pass is counted as neither.

**The outcome field still should not be published as a number a coach reads.**
It is better than it was and it is no longer worse than a constant on the
passes it decides, but it is not yet shown to carry information. That remains
a product decision and has not been made unilaterally.

### The 0.55-0.81 was not team assignment

This section previously named team assignment as the gating factor. It is
not. That figure was scored at matched passes, which measures kit clustering
*and* whether the event was attributed to the right player at once, and the
second factor dominates. Read off the kits by eye, clustering is **0.93**.

What the same exercise found instead: **a quarter of the tracks the pipeline
calls players are not players** -- stewards in hi-vis, staff in dark coats,
crowd, one advertising hoarding -- and they take **22% of the passer and
receiver slots on matched passes**. Those slots cannot be right. Grass
underfoot, movement and size all fail to separate them.

See `TEAM_ASSIGNMENT.md`. Roster contamination, not colour clustering, is the
binding constraint. Rejecting tracks that sit far from both kit centres --
kit-agnostic, so it transfers -- cuts the contaminated event slots from 22%
to 15% and lifts purity from 0.77/0.73 to 0.96/0.89 on the two hand-labelled
windows. Pooled, the outcome call now fires on 46% of passes that kept the
ball against 70% of those that lost it, where it began at 56% against 56%.
Still short of significance on 10 lost passes, and still not solved.

## Carry: the distance gate was the fault, partly

Carry was over-producing on every window -- 26 to 39 events against 16 to 25
real -- and neither merging near-duplicates nor re-timestamping helped. What
was left was the definition. SoccerNet's DRIVE is purposeful progression;
our carry was any possession spell of 0.8 s covering 2 m. Those two constants
are the whole of the distinction and neither had been swept.

Swept over 0 to 12 m on all four windows, the result is monotonic -- every
increase costs mean F1:

| CARRY_MIN_DISTANCE_M | mean F1 | windows above chance |
|---|---|---|
| **0.0 / 0.5 m** | **0.568** | **3/4** |
| 1.0 m | 0.553 | 3/4 |
| 2.0 m (was) | 0.552 | 2/4 |
| 3.0 m | 0.503 | 1/4 |
| 5.0 m | 0.394 | 0/4 |
| 12.0 m | 0.146 | 0/4 |

0.0 and 0.5 m produce identical output, so the gate does nothing below a
metre; the constant is therefore zero rather than 0.5, which would imply it
still filtered something. A 2 m requirement was excluding short drives that
SoccerNet still labels DRIVE, and removing it lifts recall from 0.72 to 0.80
at unchanged precision.

Carry at +/-1 s with the gate removed:

| window | n | P | R | F1 | p |
|---|---|---|---|---|---|
| w1 30:20 | 42 | 0.57 | 0.96 | 0.72 | **0.000** |
| w2 66:50 | 36 | 0.39 | 0.74 | 0.51 | 0.058 |
| w3 05:20 | 35 | 0.43 | 0.75 | 0.55 | **0.024** |
| Reading | 32 | 0.38 | 0.75 | 0.50 | **0.027** |

Three of four, including the independent match, which moved from 0.059 to
0.027. One window moves the other way: w2 from 0.026 to 0.058.

Carry remains **suggestive rather than established**. It sits near the 0.05
line on most windows whatever this constant is, and precision stays at
0.38-0.57. This is the best available setting for carry, not a fix for it.
Three other hypotheses -- merging, re-timestamping, and a minimum interval --
were measured and rejected before this one.

## Generalisation: a second match

Four 90-second windows now, three from Stoke City vs Huddersfield Town and one
from Reading vs Fulham. The Reading window was scored after every parameter
was fixed, on a different match, stadium, kit and camera crew.

Permutation p-values at +/-1 s, 2000 draws:

| | w1 30:20 | w2 66:50 | w3 05:20 | Reading 85:15 | combined |
|---|---|---|---|---|---|
| **pass** | 0.001 | 0.011 | 0.001 | **0.001** | 4/4, Fisher 3e-08 |
| carry | 0.001 | 0.030 | 0.059 | **0.060** | 2/4, Fisher 9e-05 |

**Pass detection is established.** It clears chance on every window including
the independent match, where precision is 0.66 and recall 0.68, and it also
clears at +/-0.25 s and +/-0.5 s there.

**Carry detection is not.** It failed at +/-1 s on the last two windows with
almost identical values, 0.059 and 0.060, and its precision on the Reading
window is 0.37 -- thirty events emitted against sixteen real, the worst of the
four. The combined Fisher statistic is significant, but that reading flatters
it: the two clear results share a match with a window that is marginal, so the
four samples are not independent. Carry is suggestive and unestablished, and
describing it as established after its first replication was premature.

Its weakness is timing rather than detection. On both failing windows it
clears comfortably at +/-2 s (p 0.000 on w3), and its absolute timing error is
0.48 s against 0.32 s for pass.

**Team attribution generalises.** 0.81, 0.76 and 0.73 on the three windows
with balanced possession, against 0.50 for no information. The Reading figure
matters most: team assignment clusters torso colour, so a different match
means different kit colours, and transferring to them is a real test rather
than a repeat. Window 1 cannot measure this -- 59 of its 60 labels belong to
one team.

## Merging near-duplicate detections

Every false positive the detector produced sat within five seconds of a real
labelled event and most within one -- 44 of 71 inside a second, none beyond
five (see `analyse_errors.py`). The detector was not inventing football that
did not happen; it was reporting one action twice. Collapsing near-duplicates
of the same type therefore attacks the fault that is actually present.

The window was swept on both windows with the value chosen on one reported
against the other, and the two event types want different answers:

| | window | evidence |
|---|---|---|
| carry | 1.0 s | chosen independently by both windows, improving both with no recall cost -- F1 0.66 to 0.67 and 0.54 to 0.58, recall unchanged at 0.84 and 0.68 |
| pass | 0.8 s | chosen by window 2, neutral on window 1. At 1.0 s window 2 recall falls 0.83 to 0.65, which is genuine quick exchanges being merged away |

A carry is an extended action, so its fragments sit further apart than a
pass's. One window for both types could not express that.

Result at +/-1 s, against the same measurement before merging:

| | before | after |
|---|---|---|
| w1 pass | P 0.70, F1 0.72, p 0.001 | **P 0.75, F1 0.72, p 0.001** |
| w1 carry | P 0.54, F1 0.66, p 0.001 | **P 0.55, F1 0.67, p 0.000** |
| w2 pass | P 0.42, F1 0.56, p 0.011 | **P 0.47, F1 0.60, p 0.002** |
| w2 carry | P 0.45, F1 0.54, p 0.030 | **P 0.50, F1 0.58, p 0.008** |

Precision improves by 0.05 on all four with recall untouched, and every
p-value strengthens. Events emitted fall from 80 to 74 and from 79 to 71.

Two values fitted against two windows is close to the limit of what this data
can justify. A third window should be used to check them, not to add a third.

## Both event types clear chance on both windows

After separating the possession timelines (see below), permutation p-values
over 2000 draws:

| group | tolerance | window 1 | window 2 |
|---|---|---|---|
| pass | 0.25 s | **0.013** | 0.081 |
| pass | 0.50 s | 0.051 | 0.123 |
| **pass** | **1.00 s** | **0.001** | **0.011** |
| pass | 2.00 s | **0.000** | **0.022** |
| carry | 0.25 s | 0.557 | **0.010** |
| carry | 0.50 s | 0.210 | **0.006** |
| **carry** | **1.00 s** | **0.001** | **0.030** |
| carry | 2.00 s | 0.079 | **0.009** |

At +/-1 s, the tolerance where this measurement has usable resolution, all
four are significant. Both window-1 results survive Bonferroni correction
across the eight tests (0.05/8 = 0.006).

At +/-1 s the detector emits 37 passes against 35 real on window 1 (precision
0.70, recall 0.74) and 45 against 23 on window 2 (precision 0.42, recall
0.83). Precision on window 2 remains the weak point.

### Why two possession timelines

Passes and carries need opposite answers to the same question: does a player
still possess the ball while it is in flight?

A carry is a continuous possession spell of 0.8 s or more, so the spell must
survive the ball leaving the ground -- ball speed is noisy enough that 21.6%
of frames cross the control gate, and releasing possession on each one chops
a dribble into fragments. A pass is the opposite: the spell has to *end*
where the ball was struck, or the pass is measured from the moment the ball
arrives, which puts both endpoints at the receiver's feet.

Serving both from one timeline was the fault behind every failed attempt at
pass precision. Releasing possession during flight fixed the pass geometry
(median travel 0.6 m -> 1.3 and 4.6 m) and took carry detection from p 0.001
to p 0.179 and from p 0.026 to p 0.896. Gap-filling preserved carries and left
passes measuring 0.6 m. Threshold changes could not resolve it because it was
not a threshold problem.

`detect_events` now builds both. Carries, recoveries, and goal and shot
attribution read the held timeline; passes and interceptions read the
flight-aware one. Pass precision moved from 0.54 to 0.70 on window 1 and 0.32
to 0.42 on window 2, emitted passes from 54 to 37 against 35 real, and carry
detection is unchanged -- which is the point.

## Earlier: carry holds, pass does not

Two independent 90-second windows of the same match, one in each half.
Window 1 is 30:20-31:50 (60 labels), window 2 is 66:50-68:20 (43 scored
labels). Permutation p-values, 2000 draws:

| group | tolerance | window 1 | window 2 |
|---|---|---|---|
| carry | 0.25 s | 0.568 | **0.009** |
| carry | 0.50 s | 0.207 | **0.009** |
| **carry** | **1.00 s** | **0.001** | **0.026** |
| carry | 2.00 s | 0.087 | **0.011** |
| pass | 0.25 s | 0.414 | 0.792 |
| pass | 0.50 s | **0.026** | 0.198 |
| pass | 1.00 s | **0.019** | 0.203 |
| pass | 2.00 s | 0.089 | 0.138 |

**Carry detection is established.** Above chance in both windows, and in
window 2 at every tolerance tested -- a broader result than window 1 gave.

**Pass detection is not.** It cleared chance in window 1 at p = 0.019 and
0.026, neither of which survived Bonferroni correction across the eight
tests, and it reaches p = 0.198 at best in window 2. A result that fails its
first replication attempt is most likely a false positive. Pass detection is
unproven, not merely weak, and the earlier description of it as a real finding
was premature.

The mechanism is visible in the counts: window 2 emits 56 passes against 23
real, for a precision of 0.09 at +/-0.25 s rising to 0.39 at +/-2 s. Recall
reaches 0.96. High recall with precision that low is close to what emitting
events indiscriminately produces, which is what the chance control exists to
catch.

`recovery` was tested for the first time in window 2, which contains one
BALL PLAYER BLOCK. Six events were emitted; one matched, only at +/-3 s
tolerance and with a 2.24 s timing error. A single label establishes nothing,
but nothing here suggests it works.

## The chance control is the result

These labels are dense: one every 1.5 seconds. A detector emitting roughly the
right *number* of events at arbitrary times therefore matches many of them by
coincidence. The first run scored F1 0.88 at +/-5 s tolerance, which read as
success until randomly placed timestamps scored 0.91.

Every figure is reported against the null of the same event count drawn
uniformly over the clip, with an exact permutation p-value over 2000 draws.

| group | tolerance | ours F1 | chance | p |
|---|---|---|---|---|
| pass | 0.25 s | 0.22 | 0.20 | 0.414 |
| pass | 0.50 s | 0.47 | 0.35 | **0.026** |
| pass | 1.00 s | 0.65 | 0.53 | **0.019** |
| pass | 2.00 s | 0.72 | 0.65 | 0.089 |
| carry | 0.25 s | 0.16 | 0.15 | 0.568 |
| carry | 0.50 s | 0.34 | 0.28 | 0.207 |
| **carry** | **1.00 s** | **0.66** | **0.45** | **0.001** |
| carry | 2.00 s | 0.69 | 0.60 | 0.087 |

Carry detection is above chance and survives Bonferroni correction across the
eight tests (0.05/8 = 0.006). Pass detection is above chance uncorrected at
0.5 and 1.0 s but does not survive that correction. One solid result and one
suggestive one.

The tolerances either side are insignificant for opposite reasons. At 0.25 s
the pipeline's timing is not that precise. At 2 s coincidence dominates --
random placement alone scores 0.60-0.65 -- leaving nothing to clear. The
measurement has useful resolution only between roughly 0.5 and 1 s.

## Recall is adequate; precision is not

At +/-1 s:

| group | truth | emitted | TP | precision | recall |
|---|---|---|---|---|---|
| pass | 35 | 54 | 29 | 0.54 | 0.83 |
| carry | 25 | 39 | 21 | 0.54 | 0.84 |
| recovery | 0 | 5 | 0 | - | - |

The detector finds most real events and invents them at a similar rate. 54
passes are emitted where 35 exist; five recovery-type events are emitted where
the window contains none. Over-emission, not missed events, is what makes the
output untrustworthy to a coach.

The recovery row is "not tested", not "scored zero": this window contains no
tackles or blocks, so the five emitted are false positives with nothing to
match against.

## Camera compensation is what makes carry detectable

Carry detection was indistinguishable from random timestamps at every
tolerance until camera motion compensation was actually applied. A carry is
defined by a player and the ball moving together, and an uncompensated pan
adds the same spurious velocity to both, destroying the test.

Compensation had never once been applied on any video in this project: the
validator was rejecting correct estimates. See `src/camera_motion.py` for the
two faults -- validating Lucas-Kanade with Lucas-Kanade, over intervals long
enough that every feature had left the frame, and a residual threshold
inherited from a magnitude test that accepted every corrupted control.

## History

Measured in August 2026, lost with the working container, and re-established
on 14 September from a rebuild. The reproduction is partial:

| | August | September |
|---|---|---|
| pass F1 @ 1 s | 0.72, p 0.001 | 0.65, p 0.019 |
| carry F1 @ 1 s | 0.65, p 0.001 | 0.66, p 0.001 |
| pass precision @ 1 s | 0.68 | 0.54 |
| events emitted | 86 | 98 |

Carry reproduces exactly. Pass is weaker and no longer survives correction.
The rebuilt possession logic is not identical to the lost version, and two
bugs in the rebuild were found by this measurement rather than by reading the
code: a hysteresis comparison against the distance at which the incumbent
gained the ball rather than where they currently are (which let a player
twenty metres away keep possession, and collapsed passes to 15), and a
constant that suppressed duplicate interception events being defined but never
wired up.

## Scope

Two 90-second windows of one 720p broadcast match. That tests whether a
result holds across different passages of play; it does not test whether it
holds across different footage. Same teams, stadium, camera hardware and
production. A second match -- Reading vs Fulham, in the same SoccerNet
archive -- is the next measurement worth making.

Nothing here transfers to 640x360 fixed-angle footage, where the resolution
gap has already been shown to break detection thresholds.

The replication did what replication is for. Pass detection at p = 0.019 on
one sample was exactly the kind of figure that fails to reproduce, and it
failed. Carry detection survived, and survived more broadly than it first
appeared.

## The metre scale was recalibrated, and these figures moved

Every threshold in this document is in metres, reached by dividing pixels by
a scale taken from median player height. That scale has since been shown to
be biased: it is the scale for motion *across* the view, and motion *into*
the view covers fewer pixels per metre, so the single number overstates
pixels per metre and every distance divided by it came out short. Measured
against a ground plane built from the camera's own rotation, by 9%, 9%, 17%
and 26% on the four windows. `src/ground_plane.py` has the derivation and
VEO_FOOTAGE.md the evidence.

The scale is now corrected, which shifts every metre threshold by that much
in effect. Re-derived at +/-2 s tolerance:

| window | pass F1 | carry F1 |
|---|---|---|
| w1 stoke | 0.849 -> **0.886** | 0.716 -> 0.687 |
| w2 stoke | 0.698 -> 0.698 | 0.692 -> **0.731** |
| w3 stoke | 0.781 -> 0.767 | 0.714 -> 0.702 |
| reading | 0.727 -> 0.679 | 0.571 -> **0.625** |

Mean pass F1 falls 0.007 and mean carry F1 rises 0.013 -- neither is a real
change, which is the expected result for a uniform rescale against
thresholds sitting on a broad plateau. The possession radius was swept from
3 to 12 m under both scales and the tuning-window mean varies by less than
0.02 across 4 to 12 m, so nothing here is balanced on the exact value.

The figures at +/-1 s elsewhere in this document have not been re-derived
and predate the recalibration. They are directionally intact -- the change
is smaller than the differences they report -- but the exact decimals are
stale.

## Shots and goals, placed from the goal instead of the centre circle

Measured on five hand-labelled windows cut around shots, with events placed
by `src/goal_placer.py` -- a goal detector for the landmark, hand-clicked
corners for the camera position, `goal_pose` for the rest.

| window | ball placed | shots | goals |
|---|---|---|---|
| stoke_1302 | 179 / 1612 (11%) | 0 of 2 | - |
| stoke_4207 | 526 / 1272 (41%) | **1 of 2**, precision 1.00 | - |
| stoke_7001 | 51 / 1548 (3%) | 0 of 2 | - |
| reading_1155 | 166 / 1530 (11%) | 0 of 1 | 0 of 1 |
| reading_2519 | 152 / 1674 (9%) | 0 of 1 | **1 of 1**, precision 1.00 |

**Shots 1 of 8. Goals 1 of 2.** Against 0 of 10 and 0 of 2 on the
centre-circle anchor. These are the first shot and goal this project has
detected that were not injected by its own test harness.

*Later: neither held.* The goal did not survive the ball-path rebuild, and
the stoke_4207 shot was the cross before it -- see "Where the camera is"
below. On the current pipeline it is 0 of 10 and 0 of 2.

The denominators differ because only five of the six windows have a label
file; the published 0 of 10 covered a sixth. The comparison is still a
comparison -- nothing here was found before -- but it is not 8 against 10 on
the same set.

### The bottleneck moved twice, and is now the ball

The anchor was the first. It is fitted from the centre circle at midfield
and a camera following play into the box does not show one, so shots were
placed nowhere at all.

Goal visibility was the second, briefly, and was my own doing: the placer
required detections at 0.35 confidence when the detector's measured
operating point is 0.15.

What remains is ball detection density, and the table says so plainly. The
only window that yielded a shot is the only one with 41% of ball positions
placed; the rest sit at 3-11%. `find_shots` fits a velocity over six placed
points within a twelve-frame window and wants three sustained hits, which at
one placement in ten frames cannot happen.

Looked at frame by frame around stoke_1302's labelled shot, the ball is
detected on 17 of 70 frames, with a 280-pixel jump between two consecutive
detections, and the strike itself has three usable points. The geometry was
right -- the ball travels from x = 13.5 m to x = 2.3 m at y = 34, dead centre
of a mouth spanning 30.3 to 37.7 -- and there were not enough points to fit
a line through.

Geometry is no longer what is failing. Better ball tracking is.

### The anchor, refitted: midfield geometry that passes an independent check

The anchor's tilt came from a horizon measured on penalty-area frames and
carried to midfield by the camera's displacement -- a step
`fit_pitch_anchor.py` itself says "treats a turn as a slide". Carrying a pose
directly also failed (`probe_rotation_carry.py`: 6 of 9 gaps broken by cuts,
6.5-20.6 m off where it crossed, against a 1.8 m floor).

`probe_constrained_anchor.py` drops the carried horizon. The goal corners fix
the camera's position per clip; a midfield frame then has four unknowns --
rotation and focal -- which the centre circle's observed arc and the halfway
line over-determine. Player sizes never enter the fit, so they test it:

| clip | pose | observed ÷ predicted player height | spread | n |
|---|---|---|---|---|
| reading_0737 | old anchor | 19.97 | 0.90 | 341 |
| reading_0737 | **refit** | **1.04** (IQR 0.96–1.09) | **0.13** | 387 |
| stoke_7001 | old anchor | 28.30 | 0.82 | 213 |
| stoke_7001 | **refit** | **1.11** (IQR 1.02–1.20) | **0.16** | 280 |

Within 4 and 11 per cent at the median, over hundreds of players the fit
never saw. The spread tests the tilt specifically: a wrongly tilted pose gets
near and far players wrong by different amounts, so the ratio wanders with
depth, and the refit's does not. The refitted focal (2657, 1943 px) sits with
the corner bundle's (3103, 1677) and the anchor's own vanishing points
(2331-3295).

**No manual midfield labelling is needed.** The fix was to remove a freedom,
not to add information.

**The poses share the goal's frame by construction** -- the circle is placed
52.5 m out from the calibrated goal line -- so the half-turn ambiguity that
closed off combining goal and midfield geometry does not arise for them.

Limits: two clips; circle frames only, not all of midfield; one round of
corner labels per clip; a 105 m pitch assumed, which EFL grounds depart from
by up to about 5 m.

Two bugs were caught on the way, both by making the synthetic check harder
rather than by reading code: a ground-space residual that let the focal
collapse to infinity and score a perfect 0.00 m, and a homography
decomposition that kept whichever of H and -H came out, landing on real
anchors with the camera looking away from the pitch.

**The camera position used above was wrong** on stoke_7001, by 25 m -- see
the next section. Rerun with the located camera the ratios are **1.01**
(IQR 0.95–1.07, spread 0.12) and **1.07** (IQR 0.99–1.11, spread 0.11):
both nearer 1 and tighter, but the table above passed with a camera 25 m
out, so this check is weakly sensitive to position and is not what
established it.

### Where the camera is: the goal's line and the centre circle's, crossed

Wired in alongside the goal, the refit kept almost nothing: 2 of 73
circles with a halfway line on stoke_7001 and 0 of 43 on reading_1155
passed the 8 px gate. The circles were not the problem -- 66 of those 73
fitted at 8 px or better **without** the halfway line. The halfway line and
the camera position disagreed.

**Goal corners fix the camera only to a line.** Focal and distance trade
against each other: on stoke_7001's one usable corner frame, assuming 1500,
2500 and 4000 px puts the camera at (-32, 10, 25), (-55, 16, 41) and
(-90, 25, 66), reprojecting at 2.1, 1.0 and 0.6 px -- nothing picks one.
The pipeline took the ground plane's focal and landed at (-36, 11, 28).
Clips with a single corner frame are exposed to this; a bundle over several
constrains it only as far as the framings differ.

**The centre circle fixes a second line.** Fitted with the camera free,
71 of 73 circles on stoke_7001 pass, every freed camera at z = 51-53 -- level
with the halfway line -- and all on one ray out of the centre spot, spread
98 m along it: the same dolly-zoom ambiguity, pointing the other way.

`src/camera_position.py` places the camera where the two lines cross,
voting over up to 30 circle frames and refusing any whose line misses the
corners' by more than 3 m. The corner poses are then refitted with that
position held (`goal_pose.pose_at`), and both placers use it.

| clip | voting | camera (x, up, z) | lines miss | spread | corners alone |
|---|---|---|---|---|---|
| reading_0737 | 29 / 30 | (-63.4, 18.4, 52.4) | 0.1 m | 0.2 m | (-52.6, 15.6, 44.0) |
| reading_1155 | 26 / 30 | (-64.1, 18.8, 52.3) | 0.3 m | 0.5 m | (-32.6, 10.7, 27.8) |
| reading_2519 | 21 / 30 | (-63.0, 19.1, 52.4) | 0.5 m | 0.2 m | (-57.8, 17.7, 48.8) |
| stoke_4207 | 11 / 11 | (76.2, 19.8, 52.7) | 1.3 m | 0.3 m | (53.5, 14.2, 36.0) |
| stoke_7001 | 21 / 30 | (-70.8, 20.3, 52.2) | 1.6 m | 0.7 m | (-36.2, 11.4, 27.5) |
| stoke_1302 | 2 -- refused; height borrowed, below | (80.0, 20.1, 53.2) | | | (33.1, 8.5, 20.2) |

Checks nothing in the fit forced:

* **Three Reading clips, fitted independently, agree to 1.1 m** --
  x -63.0 to -64.1, height 18.4 to 19.1.
* **Two Stoke clips calibrated from opposite goals agree**: 74.5 and
  72.5 m out from the goal's centreline (x is mirrored because the goal is
  the other one), 20.3 and 19.8 m up.
* Every camera sits on the halfway line, z = 52.2-52.7 against 52.5,
  where a main broadcast camera is mounted.
* Player heights, above, move to 1.01 and 1.07 and tighten.
* Against the camera heights from player sizes (`probe_anchor_height.py`):
  stoke_7001 21.9 m and stoke_4207 24.8 m, located 20.3 and 19.8, corners
  alone 11.4 and 14.2; reading_0737 15.9 m, located 18.4, corners alone
  15.6. Two of three closer, one further; that estimate fits at R² 0.16.

**What it cannot see.** The goal and the centre spot both sit on the
pitch's long axis, so a second camera displaced within the plane through
that axis and the first -- same height and distance out, further along the
touchline -- gives lines that still cross. The self-test pins that case as
accepted. Out of that plane the lines miss: 7.8 m for a camera 8 m lower,
5.0 m for one 15 m nearer the touchline, both refused.

It also inherits a 105 m pitch, which puts the centre spot at 52.5 m; on a
100-102 m pitch the camera moves 1-2.5 m along the pitch.

#### stoke_1302: a height borrowed from the same match

stoke_1302 shows the centre circle with its halfway line on 2 frames, too
few to vote. Its match (Stoke v Huddersfield, from the label file's
header) has two located clips, and a gantry camera's height does not depend
on which goal a clip calibrated from. So its camera goes on its own corner
line at their median height, 20.1 m: (80.0, 20.1, 53.2), against
(33.1, 8.5, 20.2) from the corners alone.

That is checked, not trusted: the result must land within 5 m of the
halfway line, which nothing in the construction forces. It lands 0.7 m
from it. Two other ways of choosing the point on the same line agree:
on the halfway line (79.0, 19.8, 52.5), and at stoke_4207's distance out
(76.2, 19.1, 50.5).

#### Painted lines, which no fit has seen

The strongest check, because the lines are laws-of-the-game distances and
nothing here fits to them. Clicked on the shot frames, placed through the
corner pose with each camera:

| clip | landmark | truth | corners alone | located |
|---|---|---|---|---|
| stoke_1302 | six-yard box, far corner | (5.5, 24.8) | 1.0 m off | **0.6 m** |
| stoke_1302 | six-yard box, near corner | (5.5, 43.2) | 2.0 m off | **0.0 m** |
| reading_1155 | penalty-area line, 3 points | x = 16.5 | x = 13.0, 13.9, 17.5 | **x = 16.8, 16.8, 16.8** |
| reading_1155 | penalty-area corner | y = 13.8 | y = 18.3 | **y = 13.9** |

The corners-alone camera bends the penalty line through 4.5 m; the located
one keeps it straight and 0.3 m out. Clicks are by eye on a 1280 px frame
and good to a few pixels.

stoke_1302's ball placement **falls**, 10% to 7%, and the loss is not the
camera's. 44 of the 50 lost points are one stretch, frames 619-672, where
the ball is in the air at head height and the goal is cut off at the frame
edge, so its detected box is truncated. A pose from a truncated box is
wrong whichever camera it is held at -- on that stretch the six-yard corner
lands 11 m off with the old camera and 22 m with the new -- and the new
camera is the one that puts the ball off the pitch and refuses it.
Truncated goal boxes are a separate flaw in `pose_from_box`, not addressed
here.

#### What it changes

| clip | ball placed, before | after | midfield poses, before | after |
|---|---|---|---|---|
| stoke_1302 | 10% | 7% (see above) | 0 | 2 (height borrowed) |
| stoke_4207 | 43% | 45% | 0 | 11 |
| stoke_7001 | 3% | **16%** | 0 | 53 |
| reading_1155 | 12% | **21%** | 0 | 34 |
| reading_2519 | 14% | 14% | 19 | 64 |
| reading_0737 | - | 26% | - | 85 |

xG at the six labelled shot moments barely moves -- distances shift by
0.1 to 1.6 m and the total goes from 0.49 to 0.47 (stoke_1302 16.0 to
15.4 m, 0.092 to 0.098). That is expected: near
the goal the goal's own image pins the geometry, and the camera's position
matters most far from it.

**Shots 0 of 10, goals 0 of 2.** The one shot the goal placer had matched,
stoke_4207 at 01:06, is lost, and it was not the shot. What matched, 1.7 s
before the label, is the ball crossing from the right wing, about (18, 60),
to the near post, (5, 36), in 0.7 s, after which the ball is not detected
again until after the label: the cross before the shot, inside the
matching tolerance. Its ground speed read 45 m/s with the old camera --
the plausibility ceiling exactly -- and 50-57 with the located one, above
it. Either is too fast for a ball on the grass; it was most likely in the
air, which stretches any ground projection. The ceiling is not being raised
to recover it. The earlier "1 of 8" should be read as a cross.

### What a goal box is: the placer was reading it wrong

Every goal pose away from the corner-labelled frames comes from a detected
box and the located camera (`goal_pose.pose_from_box`). It assumed the box
encloses the goal. **It does not.** The hand-drawn boxes the detector
learned from run from the top of the left post to the base of the right
post -- one diagonal of the goal's slanted outline:

| box read as | worst-edge error against the clicked corners, 25 frames |
|---|---|
| enclosing the goal | median 12.5 px; 45-62 px on every Reading frame and stoke_7001 |
| left post top to right post base | **median 3.5 px**, 24 of 25 (the exception is the replay camera) |

Where the crossbar is level the two nearly coincide, which is why
stoke_4207 and stoke_1302 were close and the others were not. The detector
reproduces the labels, so it draws the diagonal too.

The box is now read that way. Two further changes handle goals **cut off
by the frame edge**, whose box edge on that side is the frame's: that side
is scored one-sidedly (the goal must reach at least that far), roll is held
near zero -- a gantry camera does not roll, measured -1.7 to +0.2 degrees on
the corner frames -- so three real edges suffice, and a pose that would put
more than half the goal outside the picture is refused.

Measured by `probe_truncated_box.py` on real frames with the located
cameras: median metres between where a box pose and the corner pose put a
grid of ground points across the penalty area.

| | before | after |
|---|---|---|
| whole boxes, 15 corner frames | 18.44 m (90th pct 22.12) | **0.53 m (90th pct 1.05)** |
| the same boxes cut by a pretend frame edge | 31.66 m (38.35) | **1.09 m (3.54)** |
| stoke_1302 frame 638, a 34 px sliver of goal | 23-34 m off painted lines | refused |

**This retires the "1.8 m median, 3.2 m at the 90th percentile" figure**
quoted above and in several probes as the box pose's error. It was measured
against corner poses sharing the same wrong camera, so it compared two
wrong answers with each other; against the located camera the old box
reading was 18 m out. Conclusions that leaned on 1.8 m as a floor -- the
rotation-carry comparison, and out-of-play's "the margin is smaller than
the error" -- are revisited where they stand.

### Out of play, re-measured on the corrected geometry: the ball, not the line

With the camera located and boxes read correctly, placement error is about
0.5 m median (1.05 m at the 90th percentile) rather than 1.8 m, so the
argument below -- a 1 m margin asked to resolve less than its own error --
no longer holds. Re-measured on six windows:

**Recall 3 of 4, precision 3 of 15.** Barely different, which is itself the
finding. Every detected crossing, on the scorer's own placer:

| | matched (3) | false (12) |
|---|---|---|
| metres past the line | 1.7, 7.1, 7.9 | 2.0 ... 7.2, eight of them over 4 m |
| consecutive readings outside | 3, 11, 14 | 2 ... 60 |
| seconds until back inside | 0.7, 1.0, 3.1 | 0.1 ... 58 |

Still overlapping on every axis -- and the false ones are not marginal: most
sit 6-7 m past the line, where a 0.5 m placement error cannot put them.
Rendered, the four checked (reading_2519 9.9 s, reading_1155 20.0 s,
stoke_7001 41.8 s, reading_0737 60.4 s) are **not the ball**: the tracked
candidate is on the advertising hoardings or in the stand behind the line.
The geometry is right to call those points out of the pitch; the ball
tracker gave it the wrong object.

So out-of-play is now limited by ball identity near the touchline, not by
geometry or by the margin.

#### The continuity check: built and measured

A crossing now counts only if the reading before it is on the pitch,
within 0.5 s, in the same tracked segment, and near enough for a ball at
45 m/s plus 2 m of placement slack (`detect_ball_events.entered_from_inside`).
Every number is the tracker's own limit or the measured placement error;
none was chosen on the labels. Predicted before running: most hoarding
detections go, with the risk a real crossing the tracker lost on its way
out. Same placer, with and without it, six windows:

| | without | with |
|---|---|---|
| detected | 15 | 3 |
| false | 12 | **1** |
| real found, of 4 labelled | 3 | 2 |
| precision | 3 of 15 (0.20) | **2 of 3 (0.67)** |
| recall | 3 of 4 | 2 of 4 |

Eleven of the twelve false crossings go. The one new false positive
(stoke_7001, 11.4 s) is a later reading that became a fresh crossing once
the hoarding at 10.1 s was refused.

The real crossing lost, stoke_4207 at 01:07, is the **airborne ball**, not
a lost track: tracked cleanly to (4.1, 35.0) in front of goal at 65.12 s,
then read at (-7.9, 20.2) 0.24 s later -- 19 m, 80 m/s on the grass --
jumping 100 px up the image as it goes over the bar. A ball in the air
projects far beyond where it is, the same limit that turned the "shot" of
the earlier table into a cross. Loosening the speed allowance to keep it
would readmit the hoardings; it is recorded, not recovered.

Four labelled crossings still cannot validate a detector. What this shows
is narrower: the false positives were a single mechanism, and removing it
by physics rather than by threshold takes precision from 0.20 to 0.67.

The section below is kept as it was written; its conclusion about the
margin rested on the retired 1.8 m figure.

### Shots at the goal plane: 3 of 10 and the goal, where there had been none

Every labelled shot and goal was traced through the ground-based shot
detector's conditions. Sparse ball detection explains only some of the
misses -- on 10 of the 12 moments the ball is on the tracked path in 31-47 of
the 50 frames around them. What the trace found:

| cause | moments |
|---|---|
| ball in the air: 56-112 m/s on the grass, direction skewed | 5 |
| 2-5 placed positions in the 3 s around it | 4 |
| no fast movement toward goal in the window | 3 |

Rendered, reading_1155's goal is a strike on the edge of the box, no
detections in flight, and the next sighting in the top corner of the net.
Projected onto the grass, that sighting lands 2.9 m wide of the post and
makes the strike 72 m/s.

`src/goal_plane.py` reads a shot where it crosses the goal line instead. The
mouth is a vertical rectangle in the goal's frame, so a sighting's ray can
meet that plane rather than the grass, giving position across the mouth and
height at the line. A shot is a sighting whose ray crosses near the mouth
(within 3 m wide, 2 m over), from the last grass reading in shooting range
inside 1.5 s, at 10-45 m/s. All physical limits; none chosen on the labels.

| | ground detector | goal plane |
|---|---|---|
| labelled shots found | 0 of 10 | **3 of 10** |
| labelled goals found | 0 of 2 | **1 of 2** |
| false shots, six windows | 0 | **1** |

The three: reading_1155's goal, on target, 2.1 m up at the line from
17.6 m; reading_0737 at 00:20, on target, 2.0 m up from 7.9 m; and
stoke_4207 at 01:06, off target, 2.8 m up -- over the 2.44 m bar, which
agrees with the labels' out of play a second later. The false shot is on
reading_1155 four seconds after the goal, most likely the ball still in the
net; it is recorded, not ruled out by a threshold fitted to it.

Of the seven missed, four are the moments with 2-5 placed positions, which
no detector reading positions can recover. A cross passing 3 m up within
about 5 m of goal would also read as an off-target shot; the controls show
one 8 m out does not.

The scorer now uses this detector wherever the placer has a camera pose
per frame.

#### Goals at the goal plane too

The scorer's goal row came from the ball crossing the line on the grass,
which misses any goal scored in the air. A goal is now an on-target
crossing after which, within 3 s, the ball is seen in the net -- behind the
line, inside the mouth -- more often than back in play on the pitch more
than 1 m in front of it. After a goal nothing counts as a shot for 20 s,
short of any kick-off.

The first version vetoed a goal on any reading in front of the line. On
reading_1155 the ball sat in the net for half a second and the tracker
then hopped to a hoarding beyond the touchline and, for three frames, to
something by the post; the goal was called a save, and the ball still in
the net four seconds later was called the goal. Requiring in-play readings
to be on the pitch, and the net to win a majority, fixed it; a control now
pins that case.

| | before | goal plane |
|---|---|---|
| goals found | 0 of 2 | **1 of 2**, no false goal |
| shots found | 3 of 10, 1 false | **3 of 10, no false shot** |

reading_1155's top-corner goal is a goal, 0.9 s from the label, and the
false shot after it is gone under the restart rule. reading_0737's
on-target shot stays a shot. The other goal, reading_2519, has 2 placed
positions in the 3 s around it.

#### Confirmed by the kick-off

After a goal the next restart is a kick-off from the centre spot, before
anything else happens. A goal is now confirmed when the ball is seen still
on the spot (within 3 m, for 1 s) after it; refuted and downgraded to "on
target" when another shot comes first; and reported unconfirmed when the
clip ends or no kick-off is seen. reading_1155's goal is confirmed by its
kick-off inside the 90 s clip, at 85.1 s, 65 s after the goal -- checked by
eye on the frames: the ball on the centre spot, the teams in their halves,
the ball still for the two seconds rendered. Controls cover all three
verdicts.

### The hoardings: a real cause, and a fix that cost more than it gained

Three of the four sparse shot moments are not short of detections: the
detector has a ball candidate on 45-50 of the 50 frames around them, at the
stored resolution and at full resolution alike. The path was running along
the advertising hoardings on the far side while the ball was among the
players -- on stoke_1302 at 38 s the chosen candidates had a median 0.11 of
grass around them against 0.95 for those left out, at the same detector
confidence. A printed logo looks more like a ball to the detector than a
ball on grass does. (Re-detecting at full resolution helps the other two:
stoke_4207 at 66 s goes from 21 to 47 frames with a candidate.)

Weighting each candidate by the grass around it fixed the hoardings and was
reverted, because measured on all six windows it lost more than it gained:

| | before | grass-weighted |
|---|---|---|
| shots found | 3 of 10 | 2 of 10 |
| goals found | 1 of 2 | 0 of 2 |
| false goals | 0 | 1 (flagged "no kick-off seen") |
| out of play | 2 of 4 | 1 of 4 |

It gained reading_2519's shot and lost reading_1155's top-corner goal,
reading_0737's shot, stoke_4207's ball over the bar and an out-of-play
crossing. Those events happen where the ball leaves the grass background --
the net behind a goal, the stands above the bar, the hoardings past the
line -- so a preference for grass removes exactly the frames they need.
Telling a ball from a logo needs something about the object, not its
background.

### Full-resolution re-detection: tried and reverted

The clips were detected at 640 or 960 px on 1280 px footage, and at the
footage's own width the detector found a candidate on 47 of 50 frames
around stoke_4207's shot at 66 s where the stored detections had 21. So the
same detector was run at full width on every frame where the goal was in
view, its new candidates joining the stored ones. Six windows:

| | before | full resolution |
|---|---|---|
| shots found | 3 of 10, 0 false | 3 of 10, 0 false |
| goals found | 1 of 2 | 0 of 2 |
| out of play | 2 of 4, 0 false | 3 of 4, 4 false |

It recovered stoke_4207's airborne out-of-play crossing and found no new
shot; it lost reading_1155's goal and added four false crossings on three
windows. Full width finds many more candidates -- 1,364 new ones on
reading_2519 -- and more candidates are more chances to take a logo or a
boot for the ball. The limit is telling the ball from what looks like it,
which more candidates make harder rather than easier.

### A ball classifier: separates balls from logos, and learns the grass

`src/ball_classifier.py` judges the candidate itself: a crop 2.5 times its
box, a small CNN. Trained on the Roboflow football-players dataset (CC BY
4.0) -- 565 hand-labelled balls, and 667 of the detector's own look-alikes
from the same images -- and held out on three whole source matches it
scores AUC 0.956 (precision 0.81, recall 0.95 at 0.5). On stoke_1302 at
38 s it scores the hoarding candidates 0.00 and those on the pitch 0.86,
and the path leaves the hoardings.

On the six labelled windows it lost every event:

| | before | with the classifier |
|---|---|---|
| shots found | 3 of 10 | 0 of 10 |
| goals found | 1 of 2 | 0 of 2 |
| out of play | 2 of 4, 0 false | 1 of 4, 1 false |

On reading_1155's goal it scores the ball at the strike, on grass, 1.0 --
and the same ball in the top corner of the net, and lying in it after, 0.0,
with look-alikes nearby at 0.5-0.7. Nearly every training ball is on grass,
so grass became part of what a ball is, and the held-out test could not
show it because its balls were on grass too. The failure is the grass
weighting's again, learned rather than written. Reverted; the classifier
and its training script are kept unwired, since what it lacks is training
balls against the net, the stands and the hoardings -- examples these
public labels barely contain and a labelling round on this footage could.

### The camera between grid frames: interpolated, not frozen

Goal poses are solved every 4 frames and were reused unchanged in between,
so a panning camera stood still for a few frames and then jumped. Measured
without labels -- a player's feet are on the grass and cannot move more than
about 0.4 m between frames, so the frame-to-frame jitter of their placed
positions is pose noise plus the detector's own -- on every player track:

| clip | nearest pose, median / 90th pct | interpolated |
|---|---|---|
| stoke_4207 | 0.27 / 1.09 m | **0.15 / 0.47 m** |
| stoke_7001 | 0.25 / 1.17 m | **0.15 / 0.51 m** |
| reading_1155 | 0.21 / 0.84 m | **0.12 / 0.49 m** |

A still ball on stoke_4207 misfits a rolling model by 4.2 px instead of 6.2.
Stronger averaging (over 3 or 4 grid steps) and local quadratic fits add
nothing further; what remains is most likely the detector's jitter on where
the feet are, which pose smoothing cannot reach.

### Ball height: tried two ways, and neither measures these flights yet

Placing the ball assumes it is on the grass; an airborne ball projects far
beyond itself. That turned a cross into a 45-57 m/s "shot" and cost a real
out-of-play crossing. Two independent estimates of height were tried.

**From the flight's shape** (`src/ball_height.py`). Between touches a ball
follows a parabola, and gravity is a known acceleration in metres, so a run
of rays from a posed camera pins the flight in 3-D. Each window is fitted as
a flight and as a roll, and flight is believed only where it fits clearly
better, fits well near the frame asked about, and is something a football
can do (under 30 m up, under 45 m/s, near the pitch).

On synthetic flights from a located camera it works, and how long a flight
is seen decides everything:

| window | flights recognised | median height error | rolling balls |
|---|---|---|---|
| 0.48 s | 16-17 of 30 | 2.4-2.7 m | all |
| 0.8 s | 34-36 of 40 | 0.4 m | all |
| 1.0 s | 40 of 40 | 0.22-0.26 m | all |

It survives pose wobble: with per-frame camera noise of 5.5 px a flight
still misfits rolling by 9-12 px over a second, and no grass pass is taken
for a flight. The first real run exposed two faults the controls then
pinned: a window straddling a kick was "explained" as a flight 20-55 m up,
and a window half still ball and half flight let rolling pass for a ball
1.8 m up. Now such frames are left unknown.

**On stoke_4207 it identifies the dribble as rolling** (62.8-64.4 s, 3-D
position within 0.2 m of the grass placement) **and declines the cross**:
only 0.6 s of the flight is tracked before the ball is lost, under the
0.8-1 s it needs, and the per-frame box poses wobble by 6 px (a still
ball misfits by 6.1 px through them). It refuses rather than guessing,
which is right, and measures nothing here.

**From the ball's apparent size.** A ball is 0.22 m across, so its size
gives its range along the ray. Checked where the answer is known -- the
dribble, on the grass -- the detector's box is about 20% larger than that,
putting a grounded ball 2.5-4.7 m up. The bias could be calibrated out on
frames the flight fit calls rolling. What cannot be, from what is stored, is
motion blur: in flight the ball crosses about 19 px a frame and its box
grows from about 12 to 15-18 px, so size overstates height exactly on the
frames that matter.

**What would make either work.** Smoothing the camera pose over time -- a
pan is smooth, the 6 px wobble is the box fit -- would let shorter flights
be fitted. Re-detecting the ball with both box dimensions would separate
blur, which stretches along the motion, from size, which the other
dimension still shows. Neither is done; the height module is committed as
the validated part and is not yet used by any detector.

### Out of play: why it cannot be fixed on this footage

Precision is 2 of 11 and recall 2 of 3, and the attempt to improve it ended
in a negative result worth recording so it is not retried blind.

Every detected crossing was characterised three ways and compared against
the labels:

| | matched (2) | false (9) |
|---|---|---|
| metres past the line | 2.4, 7.8 | 1.8 … 7.6 |
| consecutive readings outside | 3, 20 | 3, 3, 3, 3, 4, 4, 6, 11, 13 |
| seconds until the ball is back inside | 1.3, 3.1 | 0.1 … 2.4, 7.7 |

**All three overlap.** The first was expected to separate them and does the
opposite: the falsest detection is 6.9 m out and one of the two real ones is
2.4 m, so raising the margin removes a true crossing before an invented one.

The cause is a mismatch of scales rather than a badly chosen threshold. The
margin that defines "out" is 1 m; the placement it judges is measured at
**1.8 m median error and 3.2 m at the ninetieth percentile**. The test asks
the geometry to resolve a distance smaller than its own uncertainty. Balls
near a touchline are also usually airborne -- crosses and clearances -- which
is exactly where the ground projection is worst.

A 1.25 s cut on "seconds until back inside" would lift precision from 0.18
to 0.40. It is not applied. That threshold sits between two true positives
and has false positives on both sides of it, so it is fitted to two events
and would not survive a sixth clip.

**What would actually settle it** is more labelled crossings. The SoccerNet
ball annotations carry 85 OUT events across the two matches, but only three
fall inside the ninety-second windows that exist locally. Cutting windows
around the other 82 needs the full match video, which the data agreement
does not permit holding. Until then out-of-play has three labelled events,
and three cannot validate a detector or choose a threshold for one.

### The anchor, checked against two independent camera heights

The anchor underpins possession, out-of-play and set pieces and had never
been checked against anything outside itself. `probe_anchor_height.py`
decomposes its homography into a camera pose -- inverted it is
`K [r1 r2 t]` up to scale -- and compares the implied camera height with two
estimates that use no anchor at all.

| clip | anchors | anchor | player sizes |
|---|---|---|---|
| reading_0737 | 31 | **62.2 m** | 15.9 |
| stoke_1302 | 1 | 62.7 m | 40.1 |
| stoke_4207 | 1 | 72.2 m | 24.8 |
| stoke_7001 | 31 | **70.9 m** | 21.9 |

reading_0737 carries the conclusion: 31 consistent anchors, the only height
fit worth having (R² 0.16), and the hand-clicked goal corners agreeing with
the players at **15.6 against 15.9 m**. The anchor says 62.2.

**It is not the focal.** Implied height rises monotonically with focal --
40.3 m at 1200 px, 62.2 at the ground plane's 1851, 100.8 at the corner
bundle's 3000 -- and the focal that would bring the anchor to 15.9 m is
473 px, a 107° lens on a gantry camera.

**It is not the scale.** Players placed through the same anchors span 20 m
across and 37 m along, x from 35 to 56 m, straddling the halfway line --
exactly where an anchor fitted to the centre circle should sit. Nothing
sprawls off the pitch.

Both hold because a homography with the wrong **tilt** maps the visible
trapezoid onto a believable patch of pitch while distorting depth inside it.
Perspective, not scale. The anchor returning 62-72 m across two stadiums
rather than scattering says this is systematic in its construction.

**What follows.** Anchor-derived absolute positions should not be trusted
along the viewing direction. Possession is a nearest-player comparison and
is relative, so it is largely unaffected. Out-of-play and set pieces rest on
where the ball is against a line, and are affected — which is consistent
with out-of-play never having exceeded 1 of 3.

The decomposition is verified before any clip is read: a camera of known
height projects the pitch, the homography is rebuilt from that projection
and the height recovered exactly, to 0.000 m at 15.6, 24.8 and 40.1 m. The
probe refuses to report clip numbers if that fails.

### Combining the two landmarks: measured, and refused

The goal covers the attacking third and the anchor covers midfield, so using
both should roughly triple coverage -- on reading_2519 the goal places 149
ball positions and the anchor 327. It is not available, and the reason is a
property of the footage rather than of the code.

**They never share a frame.** Not one, across reading_2519's 103 goal poses
and 117 anchor maps, both spanning the whole clip; still none allowing two
frames of slack. A tight broadcast framing shows the box or the centre
circle, never both. The nearest anchor frame to a goal frame is 59 frames
away on that clip and 812 -- thirty-two seconds -- on stoke_1302.

That matters because the two do not share a coordinate frame either. The
anchor's x = 0 may be the calibrated goal or the other one, and a wrong
choice puts every anchor-placed ball at the far end of the pitch.

**Bridging with players** -- the same track id, placed through the goal on
its frame and through the anchor on the nearest frame it covers:

| bridge | comparisons | as-is | turned | margin |
|---|---|---|---|---|
| 3.0 s | 6 | 29.3 m | 49.5 m | 20.2 m |
| 6.0 s | 25 | 40.0 m | 51.2 m | 11.2 m |

As-is wins both times, by less than the 30 m margin required, so
`goal_placer.combine` refuses and uses the goal alone.

**The residual is not evidence against the anchor.** Over three seconds a
player covers twenty metres and over six he covers fifty, so a 29-40 m
disagreement is what player motion alone produces. The measurement cannot
separate a wrong anchor from too long a bridge, and the bridge cannot be
shortened because the landmarks are never close in time here.

What would settle it is footage where both landmarks appear within a second
of each other -- a wider angle, or a full match containing such moments. Not
more code.

### What got worse

Out-of-play precision. The anchor placed so little that it reported almost
nothing; the goal placer reports more, and more of it is wrong -- five false
"out" calls on stoke_4207, two on stoke_7001. Recall is unchanged at 1 of 3.
More detections is not better detection, and this is the line of the table
that says so.

### The ceiling that is not a bug

Placing a ball assumes the ball is on the grass. A struck ball is in the
air, its ray meets the ground plane beyond where it really is, and near the
horizon a few pixels become tens of metres. Measuring the *shooter's feet*
avoids this for a known shot moment -- that is how the six shot distances in
`check_goal_corners.py` are obtained -- but a detector has to work from the
ball, because speed and heading are what make a shot a shot. The anchor
homography had the identical flaw, so this is not a regression; it is a
limit on how far this approach goes.

## Shots, goals and xG: the older position

The section above supersedes what follows, which was written when the only
labelled footage contained no shots at all. It is kept because the reasoning
about xG and held-out evaluation still holds.

Nothing in this document covers shots, because there is nothing to cover.
**All four labelled windows contain zero shots and zero goals.** Every
accuracy figure here rests on those windows, so a shot detector could not be
shown to work or caught failing. The shot and goal logic is already written
and gated off behind `absolute_pitch` at `events.py:914`; that gate is
correct and opening it would produce numbers nobody could check.

### The corpus does have shots, the video does not

| | SHOT | GOAL | all visible |
|---|---|---|---|
| Stoke - Huddersfield | 21 | 1 | yes |
| Reading - Fulham | 23 | 5 | yes |

Fifty events over roughly 100 minutes each, every one marked `visible` and
carrying the team that took it -- while only four 90-second clips were ever
cut. `build_shot_windows.py` cuts the missing test set: a window per shot,
shots within 30 s merged, plus control windows containing none, with a
manifest the existing scorer reads unchanged. Stoke gives 19 shot + 15
control windows, Reading 23 + 15.

Goals will remain weakly validated whatever is built. Six across two matches,
one of them in the Stoke match, cannot separate a good goal detector from a
lucky one. Shots at 44 are genuinely measurable; goals should be reported as
a count derived from validated shots, not as a validated metric.

### The geometry is built; the model is not

`src/shot_geometry.py` computes what every xG model takes as input --
distance to the goal centre, and the angle the goal mouth subtends -- plus
defenders inside the shooting triangle, which is the one pressure feature
available from player positions and teams. It is pure geometry, so it is
checked exactly rather than measured: `check_shot_geometry.py` verifies the
closed form against an independent vector derivation (agreeing to 1e-13 rad
over 8000 points), against cases fixed by construction, and against the
invariances the geometry must have.

The angle is an `arctan2` of two terms rather than an `arctan` of their
ratio, and that is not stylistic. Inside about 3.7 m of the goal line the
denominator turns negative, where the naive form returns this:

| distance out | correct | naive `arctan` |
|---|---|---|
| 1 m | 149.4 deg | **-30.6 deg** |
| 3 m | 101.3 deg | **-78.7 deg** |
| 11 m | 36.8 deg | 36.8 deg |

It agrees everywhere except on the best chances on the pitch, which it turns
into the worst. Two of the fixed cases exist to catch exactly that.

**The xG model is fitted, not borrowed.** Coefficients have to come from
somewhere, and writing plausible-looking ones would yield a figure that reads
like Opta and is not. xGHub (`github.com/mguti97/xGHub`, CVIU) is a dataset
rather than a model -- the repository holds a README and nothing else -- and
every shot in it carries `is_goal` beside its geometry, so `src/xg.py` fits
on it and `fit_xghub.py` runs that once into a coefficients file.

Three things about that fit are deliberate:

* **Held out by match, not at random.** Two shots from the same game share a
  pitch, a camera and often a passage of play, so splitting within a match
  lets the model see its own test conditions.
* **Unregularised.** scikit-learn applies L2 by default, which is right for a
  model only used to predict and wrong for one whose coefficients get written
  down and compared against published work. On synthetic shots drawn from
  known coefficients the default penalty moved the intercept 23% off its true
  value while the slopes stayed within 6%.
* **Calibration is reported, not just AUC.** xG is consumed as a level --
  "0.6 expected goals" is a quantity, not an ordering -- so a reliability
  table and a Brier score against the base-rate floor sit beside the
  discrimination figure.

`check_xg.py` verifies all of it on shots generated from coefficients chosen
in the test: the fit recovers them to within 5%, predicted probabilities sit
0.004 RMSE from the true ones, and the dataset's ambiguous distance
convention is correctly inferred in both directions rather than assumed.

It is a **reduced** model. xGHub's contribution is body orientation, and this
pipeline has no pose estimation, so its six orientation fields, `body_part`
and `technique` are all unavailable; the fit uses distance and goal-mouth
angle only. That should be stated wherever the number is shown.

### The fit, on the real dataset

10,580 shots over 704 matches, 1,217 of them goals -- an 11.5% conversion
rate, which is what football produces. Fitted on distance and goal-mouth
angle, holding out whole matches:

| | distance | angle | AUC | Brier (base-rate floor) |
|---|---|---|---|---|
| dataset's own split | -0.1030 | +1.5605 | 0.753 | 0.0906 (0.1019) |
| **whole matches held out** | **-0.0973** | **+1.6037** | **0.766** | **0.0859** (0.0976) |

Both are reported because the dataset's own split puts 687 of its 704
matches on *both* sides, so a model can see the teams, pitch and camera it
will be tested on. The stricter split was expected to score worse for losing
that. It scores slightly better, so on this data the leakage is not worth
anything -- worth recording, since the concern was raised before it was
checked.

AUC 0.766 is where a distance-and-angle model belongs; published models
reach roughly 0.80 using the features we do not have. Calibration on
held-out shots tracks the diagonal: predicted 0.030 / 0.052 / 0.080 / 0.128 /
0.297 against observed 0.024 / 0.042 / 0.056 / 0.158 / 0.267.

The distance convention came out of the data rather than a guess:
`goal_centre` reproduces the recorded `angle_to_posts` to **0.000 degrees**,
the perpendicular reading to 1.983.

### It has never seen a penalty

`play_pattern` in the dataset is `regular` (7210), `free kick` (2012) and
`corner` (1358). There are no penalties at all. The model puts a central
shot from the penalty spot at **0.202**, and that figure is *correct for
what it describes*: the dataset's own 81 shots from that distance and angle
convert at 0.173. It is an open-play shot from 11 m, not a penalty, and a
penalty converts around 0.76.

Anything that shows xG has to treat penalties separately or it will
under-report every one of them by a factor of nearly four. The model file
carries this in a `domain` field and `fit_xghub.py` prints it on every run,
beside a table of positions that can be judged by eye -- six-yard line 0.50,
edge of the box 0.10, 25 m out 0.04, 35 m out 0.01.

### Penalties take a fixed value, and it is assumed rather than fitted

`predict(is_penalty=...)` bypasses the geometry entirely and returns
**0.76**. Bypassing is the point: a penalty's distance and angle are the same
every time and are not what makes it convert, so routing it through a model
built on open play returns the same wrong 0.20 however it is dressed up.

That 0.76 is **not measured here, and could not be**. Nothing in any corpus
this project holds labels a penalty -- SoccerNet's ball actions are PASS,
DRIVE, HEADER, HIGH PASS, OUT, THROW IN, CROSS, BALL PLAYER BLOCK, SHOT,
PLAYER SUCCESSFUL TACKLE, GOAL and FREE KICK, and xGHub's play patterns are
regular, free kick and corner. It is the conversion rate professional
football produces, reported between roughly 0.75 and 0.80, and it is a
constant of the sport rather than of a dataset -- which is why one number is
defensible for penalties where it would not be for an open-play shot. Count
penalties and goals in any corpus that labels them and the measured number
should replace it immediately.

Two safeguards, both tested in `check_xg.py`:

* `score_shots` records `xg_source` per row -- "fitted model" or "fixed
  penalty value (assumed)" -- because those are different kinds of number
  and a bare column of probabilities hides which is which.
* A shot flagged as a penalty but taken from somewhere a penalty cannot be
  taken from is reported rather than handed 0.76. That combination is a
  detection error, and a confident number is the worst thing to put on it.

Nothing in this pipeline detects a penalty, and nothing geometric could: a
shot from the spot in open play looks identical. It takes the referee's
decision, so `is_penalty` is carried on the event rather than derived.

### Three things the dataset card gets wrong

Worth knowing before anyone else reads it: the per-shot file is
`labels_tabular.json`, not the documented `tabular_data.json`; the
`play_pattern` values are `regular`/`free kick`/`corner`, not the documented
`regular`/`set piece`/`counter`; and `distance_from_goal` is distance to the
goal centre, which the card leaves ambiguous. The loader accepts both
filenames and the convention is inferred rather than assumed.

Still blocked downstream: `shot_features_from_events` refuses to run on
relative coordinates rather than computing distance to a goal that is not
located anywhere -- which is every clip processed so far. The model is ready
before the shots are.


## The ground plane's focal length was the suspect, and the measurements acquit it

Two orthogonal vanishing points give a focal length directly, from
`(v_a - c) . (v_b - c) = -f^2`, and on conditioning it is clearly the better
estimate: interquartile spread 0.02 to 0.14 against the pan-derived 0.16 to
0.27, and larger by 1.6 to 1.8 times -- 2331, 2983, 2648 and 3295 px against
1663, 1614, 1790 and 1799.

Swapping it into the ground plane and re-running the four windows makes
everything slightly worse (`experiment_focal_source.py`):

| | pass F1 | carry F1 | split-half speed | distance-rate r | km/90 |
|---|---|---|---|---|---|
| pan focal (shipped) | **0.758** | **0.686** | **0.50** | **0.20** | **11.4** |
| vanishing-point focal | 0.747 | 0.672 | 0.47 | 0.13 | 13.6 |

Five measures, all pointing the same way, at the better-conditioned estimate.
The pan-derived focal stays.

The hypothesis it was built on -- that the focal length is what makes the
ground plane harmful -- is therefore not supported. Either the noisy estimate
is nonetheless right, or the vanishing-point constraint is biased here: it
assumes a centred principal point and square pixels, and broadcast footage is
cropped and rescaled before anyone sees it. The evidence available says only
which one the downstream measurements prefer, and they prefer the pan.


## Out of play, set pieces, and a coverage problem that is not random

Both were built on the anchor rather than on the `absolute_pitch` flag,
because that flag also switches on goal detection and feeds everything the
ground plane's coordinates. Both pass their synthetic controls: a ball
walked over each edge of the pitch is named correctly, including the case
that matters -- behind the goal line but outside the posts is out, not a
goal -- and all five restart landmarks classify correctly.

Against real labels they fail completely.

| clip | OUT found | OUT true | matched | false | throw-ins | matched |
|---|---|---|---|---|---|---|
| SoccerNet w3 | 2 | 1 | **0** | 2 | 1 | **0** |
| reading | 0 | 2 | **0** | 0 | 2 | **0** |

Not one of the three real crossings found, two invented. Set pieces follow
from crossings, so the throw-ins are 0 of 3 as a consequence rather than a
separate failure.

### Why, exactly

The ball is detected at every one of them. It simply has nowhere to stand:

    Stoke   OUT at 10.6 s : 43 ball detections within 2 s, 0 placed on the pitch
    reading OUT at  5.4 s : 74 ball detections within 2 s, 0 placed
    reading OUT at 72.1 s : 51 ball detections within 2 s, 0 placed

**Zero placed, on all three.** Not a marginal miss -- there is no pitch map
on those frames at all, so the crossing cannot be seen whatever the geometry
does.

### The part worth remembering

This is anchor coverage again, but not the version already known. Coverage
being 15% on Veo and 38-66% on broadcast has been treated as a uniform
sampling rate, as though two frames in five are anchored wherever they
happen to be. They are not.

An anchor comes from the centre circle. The centre circle is at midfield.
The ball goes out at the edges. So the frames with anchors and the frames
containing a crossing are **anti-correlated**, and the coverage that matters
for this event family is far below the average -- zero on all three
occasions here, against a headline 38-66%.

Every event tied to the edge of the pitch inherits this: out of play,
throw-ins, corners, goal kicks, and goals themselves, which happen at the
one place a midfield anchor never reaches. Shots are partly spared because
they are taken from around the box, closer to where circles are seen, which
is why an injected shot is found on every clip while a real crossing is
found on none.

Raising average coverage does not fix it. What would is an anchor that works
from the markings *at* the edges -- a goal line and a touchline meeting at a
corner, a penalty area -- which is the marking classifier's job and is
currently blocked on having footage with those markings labelled. The other
route is the wider panorama, where a single frame holds both an edge and the
centre circle at once.


## Out of play and set pieces: what fixed it, what did not, and where it stands

The earlier section on this page blamed anti-correlated coverage, and that
was right about the cause and wrong about the remedy. Two remedies were
tried.

### Carrying the map further does not work

A frame with no centre circle inherits its map from a frame that has one.
That borrow was matching the anchored frame *directly* against the target,
one fit across the whole gap, so it failed as soon as the camera had panned
away -- which is why raising the limit from 6 s to 16 s helped a little and
36 s helped no more. Time was a proxy; overlap was the limit.

Carrying the map in small steps instead, composing warps between consecutive
frames, fixes the overlap problem and introduces a worse one. Measured
against anchors the carry never sees:

| gap | direct | chain/4 | chain/12 | chain/25 | chain/50 |
|---|---|---|---|---|---|
| 0-2 s | 75% 0.9m | 92% 3.2m | 92% 1.4m | 83% 1.2m | 83% 1.2m |
| 2-8 s | 62% 0.7m | 91% 9.3m | 75% 5.9m | 62% 3.1m | 62% 1.5m |
| 8-20 s | 31% 4.5m | 84% 21.2m | 67% 12.8m | 36% 6.1m | 36% 3.2m |

Drift counts **multiplications**, not seconds: the same twenty seconds costs
3.2 m in ten steps and 21.2 m in a hundred and twenty-five. Coarse steps
drift less and break more (45 of 62 fitted against 567 of 576), and one
break ends a chain, so accuracy and coverage trade against each other
directly. A hybrid that takes coarse steps where they fit and fine ones only
to bridge a break behaves exactly as designed and still loses: at 2-8 s it
buys four points of coverage for twice the error.

**The direct carry is 0.7-0.9 m out to eight seconds and 4.5 m beyond it.**
That is the number to remember, because out of play is judged against a 1 m
margin.

### More anchors does work

Only 80 of 2250 frames were ever tried for a centre circle. That is a
sampling budget, not a property of the footage. Tripling it:

| clip | anchors | placed | labelled crossings seen |
|---|---|---|---|
| Stoke, 80 frames | 19 | 749/1811 | 0 of 1 |
| Stoke, 240 frames | 47 | 1016/1811 | **1 of 1** |
| Reading, 240 frames | 60 | 1055/1820 | 0 of 2 |

The crossing that three sessions of work on carrying could not reach appears
immediately. The budget is now 240 and it is the parameter that decides
whether this family of events is visible at all.

### Two of my own claims were wrong and are withdrawn

**The 150-to-400 reach change was not what found the crossing.** At 240
anchors the crossing is found at reach 150 too. Holding the anchors fixed
and varying only the reach, it buys 169 placements and one extra false
crossing. It is not harmful -- zero false shots on five verified shot-free
clips, injected shots recovered within 0.3 m -- but the explanation
published with it was wrong. The original sweep behind that change reported
993 placements where the shipping detector reports 749 and 751; it was
written as a shell heredoc and cannot be audited, which is why
`probe_borrow_reach.py` is now a file.

**The Veo xG improvement was not a coverage improvement.** Veo's placement
is essentially unchanged (505 against 514). The injection takes the first
anchored run long enough to hold a shot, and at the longer reach an earlier
one qualified, so the shot was planted somewhere better covered. What the
0.106 against a true 0.103 shows is that Veo's geometry is sound where a
whole flight is visible. It does not show that the reach made flights
visible.

### Where out of play actually stands: 1 of 3, with 4 to 6 false per 90 s

| | found | labelled | matched | false |
|---|---|---|---|---|
| Stoke | 7 | 1 | 1 | 6 |
| Reading | 2 | 2 | 0 | 2 |

The two failure modes are different and the per-event listing separates
them.

**The misses are not geometry.** Both Reading crossings have **0 ball
positions placed** within two seconds, out of 74 and 51 detections. There is
no map on those frames, at 58% placement across the clip. Anti-correlation
survives tripling the anchors.

**The false alarms are mostly the anchor's own error.** Of the seven
spurious crossings, four sit 1.2 to 1.9 m past the line -- inside what the
anchor is known to be worth, 0.7-1.1 m anchored and several metres carried,
against a 1 m margin. Two are 11.5 and 21.2 m out, which is a broken map or
a ball detection on something that is not the ball.

Sweeping the margin from 1 m to 5 m and requiring up to three consecutive
readings never gets below four false crossings and never above one match.
**No setting rescues this**, which is the useful thing the sweep says --
and with three labelled crossings in existence, no setting could be chosen
on them anyway.

### What would fix it

The same thing as before, and the measurements have now ruled out the
alternatives. The misses need an anchor that works *at the edge of the
pitch* -- a goal line meeting a touchline, a penalty area -- because that is
where these events happen and a midfield circle never sees them. The false
alarms need a map good to well under a metre at the moment of crossing,
which is the same requirement.

### Set pieces, at the new budget: 1 of 3, for exactly the same reason

A restart is only looked for after the ball has been seen to leave, so the
two measurements are one measurement. Scored against the THROW IN labels
inside these clips:

| clip | crossings found | restarts | named | throw-ins labelled | matched |
|---|---|---|---|---|---|
| Stoke | 6 | 6 | 4 free kick, 2 throw in | 1 | **1** |
| Reading | 2 | 2 | 2 free kick | 2 | 0 |

Stoke's throw-in is the first real set piece this pipeline has ever named
correctly, and it appeared the moment the crossing before it became
visible. Reading's two are missed because Reading's two crossings are
missed.

The classifier itself is not the problem: it names all five kinds correctly
from position on synthetic input, and it is right about the one real
restart it can see. It has almost nothing real to name. "Free kick" in that
table is a residue rather than a detection -- it is what is left when a
restart matches no landmark, and nothing in a ball track shows a foul.

So both families sit at 1 of 3, and they will move together. Whatever makes
a crossing at the edge of the pitch visible makes its restart visible too.


## Possession share: the number the report prints, measured at last

Possession is the most prominent figure in the report and had never been
compared with anything. It can be: every one of the 1844 ball actions in a
labelled match carries a `team` field, so the labels are not a list of
events but a possession timeline sampled about once a second. The team in
control between one action and the next is the team of the earlier action,
and from an OUT or a GOAL until the next action nobody is in possession.

Two numbers, because the share alone cannot be trusted. **Share** is what
the report prints. **Per-frame agreement** is whether the right team is
named moment to moment, and the bar it has to clear is the majority-class
baseline: on football where one team holds 60% of the ball, always naming
that team scores 60%.

| clip | rule | frames | agreement | majority | share ours | share true | error |
|---|---|---|---|---|---|---|---|
| Stoke | proxy | 1343 | 0.69 | 0.52 | 0.35 | 0.52 | **0.17** |
| Stoke | spells | 1623 | 0.71 | 0.53 | 0.41 | 0.53 | **0.12** |
| Reading | proxy | 813 | 0.57 | 0.57 | 0.55 | 0.57 | 0.02 |
| Reading | spells | 1120 | 0.50 | 0.56 | 0.56 | 0.56 | 0.00 |

### The two clips say opposite things, and that is the finding

On Stoke the timeline **knows who has the ball** -- 0.69 and 0.71 against a
0.52 baseline, the clearest signal in this table -- and the share is out by
seventeen points. On Reading the share is **exact**, 0.00 error, and the
timeline is worth nothing: the proxy lands precisely on the majority
baseline and the spell rule scores 0.50 against a baseline of 0.56, which is
worse than naming one team and never changing your mind.

So accuracy of the share is anticorrelated with whether the thing underneath
it works. A reader given "Team A 54%" cannot tell which of these two matches
they are looking at, and neither can the pipeline.

### Why a share can be exact while the timeline is a coin flip

The share is not computed over the match. It is computed over **the frames
where the rule fires** -- where the ball is detected and a player is within
3 m of it -- and those frames are not a random sample of the football. The
proxy scores 1343 frames of 2250 on Stoke and 813 on Reading. A team that
plays long balls spends its possession with nobody within three metres of
the ball, so its possession is disproportionately invisible, and the ratio
is taken over what is left.

That is a selection effect, not a measurement error, and it explains both
rows: it can distort a share badly while each individual frame is decided
correctly, and it can leave a share intact while the frames are guesses.

### What this does not say

Two matches. The spell rule beats the proxy on Stoke and loses to it on
Reading, so nothing here picks a winner between them, and the report's use
of the weaker-looking rule is not established as a mistake. What is
established is that the printed percentage has been out by seventeen points
on labelled football, and the report now says so next to the number instead
of calling it a proxy and leaving it there.

Reproduce with `python check_possession.py`, controls with `--check`.


## What the possession share was actually wrong about: the ball in flight

Three hypotheses, measured in order. Two were wrong, which is why they are
here.

### Not a selection effect

The share is computed over the frames where the rule fires, and those are
not a random sample, so the first guess was that a team playing long balls
spends its possession with nobody near the ball and is under-counted.
Counting intervals instead of frames -- bridging the gaps between firings,
the way the truth timeline itself is built -- tests that directly:

| clip | rule | sampled | 1 s | 2 s | 5 s |
|---|---|---|---|---|---|
| Stoke | proxy | 0.17 | 0.20 | 0.19 | 0.19 |
| Stoke | spells | 0.12 | 0.11 | 0.11 | 0.11 |
| Reading | proxy | 0.02 | 0.03 | 0.02 | 0.01 |
| Reading | spells | 0.00 | 0.08 | 0.08 | 0.10 |

Coverage rose sharply -- 1343 frames to 1971 on Stoke, 813 to 1662 on
Reading -- and the share error did not move. **Hypothesis withdrawn.** The
interval counting stays because it makes the comparison like against like,
but it is not the fix.

### Not detection either

A share can only be wrong one way if the mistakes run one way, so the next
question was per-team recall:

| clip | rule | hit left | hit right | gap | share error |
|---|---|---|---|---|---|
| Stoke | proxy | 0.54 | 0.86 | 32 pts | 0.17 |
| Stoke | spells | 0.62 | 0.82 | 20 pts | 0.12 |
| Reading | proxy | 0.60 | 0.52 | 8 pts | 0.02 |

The bias is real and its width tracks the share error. The obvious cause
would be seeing one team less, so both teams were counted:

    Stoke     team_A 6.7/frame in 96% of frames   team_B 6.1/frame in 100%
    Reading   team_A 5.5/frame in 92%             team_B 4.4/frame in  96%

Near enough balanced, and the team whose possession is missed more is the
one seen *more*. **Hypothesis withdrawn.** The nearest player is not being
missed; it is being read correctly and is the wrong player.

### The ball in flight

While a pass travels, the nearest player is routinely an opponent -- the
defender it passes, the man it is played away from. The labels disagree: a
PASS belongs to the team that struck it until the next action, flight
included. That is asymmetric in exactly the way the numbers are, because it
costs whichever team plays the longer balls. It also explains why the spell
rule, which holds the ball through its flight, already had a recall gap
twelve points narrower than the proxy, which has no notion of flight.

Keeping only the frames where the ball is slow enough to be held:

| clip | rule | frames | agreement | majority | hit left | hit right | share error |
|---|---|---|---|---|---|---|---|
| Stoke | proxy | 1343 | 0.69 | 0.52 | 0.54 | 0.86 | 0.17 |
| Stoke | **settled** | 645 | **0.87** | 0.51 | 0.81 | 0.93 | **0.06** |
| Stoke | settled/1 s | 1164 | 0.82 | 0.55 | 0.76 | 0.86 | **0.03** |
| Reading | proxy | 813 | 0.57 | 0.57 | 0.60 | 0.52 | 0.02 |
| Reading | **settled** | 369 | 0.61 | 0.58 | 0.62 | 0.60 | 0.06 |

On Stoke this is the largest single improvement in the table: agreement from
0.69 to 0.87 against a 0.51 baseline, the recall gap from 32 points to 12,
the share error from 0.17 to 0.06. On Reading it closes the recall gap from
8 points to 2 and lifts a timeline that sat exactly on its baseline to
slightly above it, while moving the share error from 0.02 to 0.06 -- on a
timeline that was never informative, so there was no accurate share there to
protect, only a lucky one.

**The rule now ships gated.** Not because of those scores but because you
cannot possess a ball flying past you, which was true before anything was
measured; the scores are disclosed so the distinction between a principle
and a fit can be checked. The report prints the measured error beside the
number.

### The gate was measured in one place and shipped in another

Worth recording, because it nearly published a number that was never true.
The gate was developed in the scorer and then added to `stats`, leaving two
implementations that differed in how they smoothed the ball's speed. Scoring
the shipped function rather than a copy of it caught the gap at once:

| clip | rule | frames | agreement | hit left | hit right | share error |
|---|---|---|---|---|---|---|
| Stoke | measured here | 645 | 0.87 | 0.81 | 0.93 | 0.06 |
| Stoke | shipped, before | 960 | 0.74 | 0.63 | 0.89 | **0.16** |
| Stoke | shipped, after | 645 | 0.87 | 0.81 | 0.93 | 0.06 |

Same intent, same threshold, different smoothing. The looser estimate kept
960 frames where the other kept 645, so flight frames survived the gate and
the bias survived with them -- a 26-point recall gap against 12.
Documenting 0.06 while shipping 0.16 would have been worse than not
measuring at all.

Both now call `ball_tracking.kinematics`, and the scorer calls the shipped
`stats.possession_timeline` rather than reimplementing it, so this number
moves if the rule does. The final reading is **0.06 share error on both
clips**, from 0.17 and 0.02.

### What is still not known

Two matches. The share error is 0.06 on both after the fix, which is better
than 0.17 and is not good. The denominator also shifts between rules -- the
settled rule scores 645 frames where the proxy scores 1343 -- so the truth
share itself differs slightly between rows, and these are not four readings
of one quantity.


## Why two of the three crossings are invisible: the ball is not in the picture

Every attempt on this family has been an algorithm. Looking at the frames
took ten minutes and ended the argument.

**Stoke, 10.6 s.** The camera is zoomed hard onto the touchline, which runs
bright and unbroken across the frame. There is no centre circle and no
penalty area, so neither the shipped anchor nor a box-based one could work
here -- but the touchline itself could hardly be clearer. Also visible, and
fatal to the grass idea: **the grass continues past the touchline.** The
run-off is grass, so a ball a metre out is still over grass. That, not
lofted balls, is why `grass_frac` failed on this clip.

**Reading, 72.1 s.** Players are standing still, staff are on the touchline,
the referee is walking. It is plainly a dead ball. There is **no ball in the
frame**. The ball track's one detection within a second of it, at confidence
0.41, sits on a player's fluorescent boot; at the labelled moment and a
second later there are no ball detections at all.

So the answer to "why is the crossing not detected" is that the ball is not
visible when it leaves the pitch. A broadcast camera follows the ball, and
at the moment it crosses, the ball is at or beyond the edge of frame. No
anchor, no marking detector and no amount of geometry reaches that. It is a
property of the footage.

This also disposes of the edge-anchor plan for these cases. An anchor fitted
from markings at the edge of the pitch is still a way of saying **where the
ball is**, and there is no ball to place.

### What is visible instead

In that same frame: twenty-two players who have stopped running. A stoppage
is written all over the players even when the ball is gone, and player
tracks do not depend on the ball being in shot.

That is a different signal, available exactly where this one fails, and it
is worth measuring on its own terms -- which is the next thing here.

### And a separate finding: perfect team assignment does not help possession

Non-player rejection sits at 0.80, and the suspicion was that stewards
standing on the touchline take possession from real players. The hand-read
kit labels answer it directly, since `run_pipeline` accepts a team
override: give it truth, and non-players get no team at all.

| Reading | frames | agreement | majority | margin | share error |
|---|---|---|---|---|---|
| assigned teams | 369 | 0.61 | 0.58 | +0.03 | 0.06 |
| hand-read teams | 349 | 0.64 | 0.61 | **+0.03** | 0.09 |

The margin over the baseline does not move and the share gets slightly
worse. Whatever limits possession on this clip, it is not the roster. One
clip carries kit labels, so this is a single reading -- but it is the clip
where possession is weakest, which is where the effect should have shown.


## Stoppages from player motion: measured, and it does not work either

If the ball is not in the picture, the players are. Median player speed per
frame, a stoppage being where it drops and stays down:

| clip | best setting | matched | false | delays |
|---|---|---|---|---|
| Stoke | any of nine | 0 of 1 | 1-9 | +4 / -2 |
| Reading | still < 0.8 m/s, 1 s | 1 of 2 | 1 | +13, -0 |

The criterion was set before the run: play stops when the players notice
rather than when the ball crosses, so a working detector shows a consistent
positive delay and a coincidence shows scatter. The delays are +13, -4, -53,
-1. That is scatter.

**Every route to this event family has now been measured and rejected:**
carrying the anchor further, carrying it in steps at four spacings, a
multi-scale walk over both, tripling the anchor budget (which fixed one of
three), the grass around the ball, and now collective player motion.

## The footage is the limit, and there is no better copy of it

The clip called "Veo" throughout this repository is a **640x360 auto-follow
crop** -- a broadcast-style feed with a scoreboard burnt into the corner,
panning to keep the ball centred. It is not a Veo panorama, and there is no
panorama to fetch: it was downloaded from the web, and there is no camera,
account or original recording behind it. The name is a misnomer this
repository carried for a long time.

That matters more than anything else on this page, because the failure this
section documents is a *property of a camera that follows the ball*: when
the ball leaves the pitch the camera has not caught up, so the ball is at or
past the edge of frame and there is nothing to detect. A fixed wide camera
covering the whole pitch does not have that failure mode.

It would also remove most of what this project has spent its time on:

  * **the anchor problem disappears.** A camera that does not move has one
    homography for the whole match. Everything about carrying a map from
    frame to frame, the 6 s and 16 s reaches, the drift that counts
    multiplications, the anti-correlated coverage -- all of it exists
    because the camera pans;
  * **the ball gets more pixels.** At 640x360 the ball is about 7.8 px
    across, which is why its detection is 73% and its speed noisy enough to
    need a least-squares fit over six frames;
  * **out of play becomes observable**, because the touchline and the ball
    are in the same frame at the moment that matters.

So the recommendation is not another detector, and it is not the panoramic
export either, because there is not one. What is left is smaller and still
worth doing: **label the clip that exists.** Three minutes of it, by hand,
gives this project its first labels on footage that is not SoccerNet, and
answers the question the plausibility checks cannot -- whether the pipeline
transfers at all, or whether every number here belongs to 720p broadcast
and nothing else.

The blind spot will remain, because it belongs to the camera. Out of play
will stay unmeasurable on this footage whatever is labelled. Everything
else -- shots, possession, teams, passes -- becomes measurable where it
currently is not.


## Shot recall, measured at last: 0 of 6, and the anchor is why

The clips that existed bracketed every labelled shot without containing one,
so the shot detector shipped numbers with its recall unmeasured. Cutting
windows around the labels fixed that: three windows of Stoke -
Huddersfield, two labelled shots each.

| window | ball placed | shots labelled | found | matched |
|---|---|---|---|---|
| 13:02 | 984/1612 (61%) | 2 | 0 | **0** |
| 42:07 | 173/1272 (14%) | 2 | 0 | **0** |
| 70:01 | 939/1548 (61%) | 2 | 0 | **0** |
| **total** | | **6** | **0** | **0.00 recall** |

Zero false positives as well, which is the one consolation and the same
figure the shot-free footage gave. A detector that never fires scores that
too.

The two 61% windows are the ones to read: coverage is not the excuse there,
and it is better than the windows where an injected shot came back within
0.3 m. The 14% window fails for the ordinary reason, no map at all.

The first window is traced below.

    event      labelled   found   matched   recall   precision
    shot              2       0         0     0.00           -
    out               2       3         0     0.00        0.00

**Zero.** The detector did not fire at all, and this is not a coverage
excuse: 984 of 1612 ball positions were placed, 61%, the best this pipeline
has managed and better than the windows where an injected shot came back
within 0.3 m.

### What actually rejected the shot

At the labelled shot the ball is placed at **x = 46-63 m** -- the middle of
the pitch -- while the video shows it struck a dozen metres from goal. Its
speed reads 23.6 m/s, comfortably over the 13 m/s a shot needs. So the
detector behaved correctly on the input it was given: a fast ball in
midfield travelling nowhere near a goal is not a shot. The map underneath it
was wrong by about forty metres.

### Why the map was wrong, traced

1. The camera is zoomed into the penalty area for the whole passage, so the
   centre circle is off screen. Of 240 frames tried, **6 produced an anchor
   at all**.
2. The shot sits at frame 500. The two nearest anchors are frames 178 and
   667, and 667 is nearer, so the shot borrows its map from 667.
3. Frame 667 shows the penalty box. Re-running the circle detector on it
   with twenty different random draws finds a circle **zero times**. The
   anchoring run got one fit, and it does not reproduce.

So a single stochastic mis-fit, on a frame with no centre circle in it,
became the nearest anchor to the shot and displaced it by forty metres.

### The lesson, which is not about shots

The circle fit is RANSAC, so it is a sampler, not a function. A rare
spurious fit is not an anomaly to be surprised by; it is what a sampler does
occasionally. Nothing downstream asks an anchor to prove itself, and on a
clip where only six frames anchor, one bad anchor contaminates a large share
of the borrowed maps.

The cheap defence is to make an anchor reproduce before it is trusted: fit
twice with independent draws and keep it only if both agree. On this clip
that rejects frame 667 (0 of 20 refits) and keeps frame 178 (refits
immediately). It costs a second fit per candidate frame and some yield,
which has to be measured before it ships -- yield on this footage is already
6 of 240.

### And this changes what the earlier shot evidence meant

The injected-shot control recovered a planted shot on every clip to within
0.3 m. It did that by planting the shot **through the anchor of a frame
that had one**, which quietly guaranteed the map was good. It measured the
geometry and the xG model, and it could not have caught this, because the
failure is an anchor that should never have existed.


## Making an anchor prove itself: measured, and it ships

The gate: fit the frame twice with independent draws and keep the anchor
only if the two maps place the ground within 2 m of each other. Seeded per
frame from a fixed salt, so the second opinion does not disturb the run's
main random stream -- taking the seed from that stream made the gated and
ungated runs explore different RANSAC sequences everywhere downstream, and
the first measurement of this was void because of it. The guard that caught
it was the gate reporting an anchor it had *added*, which it cannot do.

### What it keeps

| clip | anchors, plain | gated | kept |
|---|---|---|---|
| Stoke 13:02 | 6 | 1 | 17% |
| Stoke 42:07 | 7 | 1 | 14% |
| Stoke 70:01 | 47 | 31 | 66% |
| SoccerNet w1 | 91 | 86 | 95% |
| SoccerNet w2 | 32 | 25 | 78% |
| SoccerNet w3 | 47 | 34 | 72% |
| reading | 60 | 54 | 90% |
| Veo | 13 | 11 | 85% |

The split is the finding. Windows looking at midfield keep 72-95%. The two
windows cut around shots, where the camera is zoomed into the box, keep 17%
and 14%. The anchors near the box were not occasionally wrong; they were
mostly unreproducible, and the 61% ball placement on the 13:02 window rested
on maps that cannot be found twice.

That also corrects something said earlier on this page. Frame 178 was called
a good anchor because it refit once. Over eight independent draws it fits
twice and those two fits place the ground **7.3 m apart**; frame 667, the
one that cost the shots, fits zero times. The difference between them is 2
of 8 against 0 of 8, not sound against broken.

### What it costs, and what it buys

On the shot window, placement falls from 984 of 1612 (61%) to 162 (10%) and
shot recall stays 0 of 2 -- but all three false crossings disappear. The
gate converts confidently wrong output into honest silence.

On the clips where the anchor works, it does something better than break
even. Every injected shot is still recovered -- 15.8 to 16.2 m against 16.0
planted, xG 0.101 to 0.105 against 0.103 -- and **coverage rises on four of
five clips**:

| clip | placed, plain | gated |
|---|---|---|
| w1 | 1042 | 1017 |
| w2 | 680 | 700 |
| w3 | 751 | **976** |
| reading | 873 | **908** |
| Veo | 505 | **682** |

Worth understanding rather than just noting. A bad anchor captures the
frames nearest to it as their borrow source, and then the maps carried from
it are implausible and get thrown away. Removing it lets the next-nearest
sound anchor serve those frames instead. The gate was not only removing
noise; it was unblocking coverage that a bad anchor had been holding.

**Shipped on by default**, against a rule fixed before the number was seen:
pass the injected-shot control and it ships. `--no-reproduce` restores the
old behaviour for comparison.

### What it does not do

Shot recall is 0 of 6 with the gate and 0 of 6 without it. This improves
what the pipeline says about the penalty area; it does not improve what it
can see there. That still needs an anchor built from markings visible in the
box, which is the same structural gap out of play keeps arriving at from the
other direction.


## The full shot and goal measurement: 0 of 10 and 0 of 2, on two matches

Six windows cut around labelled shots, three from each of two matches. The
Stoke windows were scored before the reproducibility gate shipped, the
Reading windows after it, which is why the false-alarm column differs.

| window | ball placed | shots | goals | found | false out |
|---|---|---|---|---|---|
| Stoke 13:02 | 984/1612 (61%) | 2 | - | 0 | 3 |
| Stoke 13:02, gated | 162/1612 (10%) | 2 | - | 0 | **0** |
| Stoke 42:07 | 173/1272 (14%) | 2 | - | 0 | 0 |
| Stoke 70:01 | 939/1548 (61%) | 2 | - | 0 | 0 |
| Reading 07:37 | 455/1651 (28%) | 2 | - | 0 | 0 |
| Reading 11:55 | 170/1530 (11%) | 1 | 1 | 0 | 1 |
| Reading 25:19 | 355/1674 (21%) | 1 | 1 | 0 | 1 |
| **total** | | **10** | **2** | **0** | |

Goal detection had never seen a real goal before these two windows. It has
now seen two, and found neither. It remains correct on synthetic input,
which is what it was always measured on.

### Placement is not the discriminator, and that is the point

Two windows place 61% of the ball, one places 11%, and the outcome is
identical. On the 61% windows the ball is placed plentifully and placed
**wrongly**: at the Stoke 13:02 shot it sits at x = 46-63 m, midfield, while
the video shows it struck a dozen metres from goal.

So this is not a coverage problem with a coverage fix. Wherever the camera
goes to watch something worth measuring, the anchor's only landmark -- the
centre circle, at the halfway line -- leaves the frame. What is left to fit
is whatever happens to be in the box, and the reproducibility gate measures
how little that is worth: windows looking at midfield keep 72-95% of their
anchors, the windows cut around shots keep 14-17%.

### What this settles

The pipeline **cannot measure anything that happens in the penalty area on
broadcast footage**. That is now a measurement on 10 shots and 2 goals
across two matches, not an inference from one clip.

It also bounds what the xG model is worth in practice. The model itself is
sound -- `check_xg.py` recovers known coefficients, and an injected shot
comes back to 0.3 m with xG within 0.007 -- but it has never been handed a
real shot by this pipeline, and on this evidence it will not be until the
anchor can work from markings inside the box.

### What still works, measured on the same runs

The same six clips scored passes and carries against the full label file.
Pass clears chance on the new windows as it did on the old ones, and the
reproducibility gate -- which changed the anchor path underneath everything
-- did not degrade it. Those detectors do not use the anchor.


## The penalty-area anchor: blocked on finding the paint

The obvious fix for 0 of 10 shots is an anchor fitted from the markings that
are in the box -- goal line, six-yard box, penalty area, and the sides
joining them. Before building a fitter, a cheaper question: are those
markings detectable in the frames where the centre-circle anchor fails?

The triage said 0-18% of frames usable, which reads as "the markings are not
there". **That answer is wrong and the reason matters more than the answer.**

Drawing the detected lines on a frame settles it. The frame shows the goal
line, the six-yard box, the penalty area and the arc clearly. What the
detector returns is mostly the **hoardings**, the crowd boundary and long
diagonals across bare grass, while the paint goes mostly untraced. Eroding
the grass mask by 31 px removes the worst hoarding lines and does not fix
the rest.

A second fault compounded it. Lines parallel to the goal line arrive at 7,
17, 19, 26, 174, 175 and 176 degrees on one frame -- a 30 degree spread,
because perspective makes parallel lines converge -- against a 25 degree
threshold for calling two lines transverse. The within-family spread is
wider than the between-family test, so the split is meaningless. Families
have to be found by vanishing point, not image angle.

### What this establishes

Not that the box cannot be mapped. That the **markings cannot currently be
found**, which is a different and more tractable problem, and one this
project has already met: the marking-classifier work was abandoned earlier
because anchor-derived training data contained zero penalty arcs -- anchors
come from the centre circle, so nothing near the box was ever labelled.

The same circularity is still in place. The anchor cannot see the box, so it
cannot label the box, so nothing learns to find the box.

### The honest next step

Break the circle with data rather than cleverness. Label the markings on a
few hundred frames of box-zoomed footage by hand -- goal line, six-yard box,
penalty area, arc -- and train a segmenter on that. `line_segments` finds
contrast; what is needed is something that finds paint, which is a
recognition problem rather than an edge-detection one.

Until then, the penalty-area anchor cannot be built on what this pipeline
can see, and the 0 of 10 stands.
