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

Two candidate causes are ruled out:

* **Team does not flicker within a track.** Zero of 646 tracks across the four
  windows ever change team.
* **It is not fragmentation.** Zero intercepted passes involve a passer and
  receiver whose tracks never coexist, so they are genuinely distinct players,
  6 to 11 m apart.

Team assignment error is a major contributor but provably not the whole story.
If each end of a pass is assigned correctly with probability a, the reported
turnover rate is

    0.88 * 2a(1-a)  +  0.12 * (a^2 + (1-a)^2)

which peaks at 50% when a = 0.5. Three of the four windows exceed that, so
something beyond team assignment is inflating the rate.

The requirement this sets is worth stating plainly: for the reported rate to
fall below 20% against a true 12%, per-end team accuracy must reach about
0.95. It currently measures 0.73 to 0.81. That is a different order of
problem from a threshold.

The remaining contribution is not isolated. Intercepted passes coincide with
two to three times higher ball speed on three windows (112 against 35 km/h on
w1) and the reverse on the fourth, and the test is compromised because
`_ball_kinematics` clips speed at 120 km/h, hiding exactly the spikes measured
elsewhere. Suggestive, not shown.

Until per-end team accuracy is far higher, the honest options are to suppress
the outcome field rather than publish a coin flip, or to report only the
aggregate with its measured error. Neither is done yet.

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
