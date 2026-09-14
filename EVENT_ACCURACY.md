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

One 90-second window of one match, 720p broadcast. Replication on a second
window and a second match is the next measurement, and it matters: p = 0.019
on a single sample is exactly the kind of figure that fails to replicate.
Nothing here transfers to 640x360 fixed-angle footage, where the resolution
gap has already been shown to break detection thresholds.
