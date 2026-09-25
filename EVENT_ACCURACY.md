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

## Shots, goals and xG: where they stand

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

## The footage is the limit, and it is the wrong footage

The clip called "Veo" throughout this repository is a **640x360
auto-follow crop** -- a broadcast-style feed with a scoreboard burnt into
the corner, panning to keep the ball centred. It is not a Veo panorama.

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

So the recommendation is not another detector. It is to run this on the
panoramic export at full resolution, and to label a few minutes of it, so
that the numbers are measured on the footage the product actually has
rather than transferred from two broadcast matches on faith.
