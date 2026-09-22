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
