# The pipeline on Veo footage

Every accuracy figure in this project comes from SoccerNet or Roboflow, both
720p or 1080p broadcast. The footage the product is for is a Veo camera at
640x360. This is the first run on it since the rebuild.

Clip: 3 minutes from 20:00 of a full match, 640x360 at 25 fps, 4500 frames.

**There are no labels for this footage**, so nothing here is precision or
recall. These are plausibility checks against what football is -- roughly 22
players, one ball, human speeds. They catch a pipeline that is badly wrong,
not one that is subtly so, and this project has been misled by exactly this
kind of check before: "100% ball detection" was a cap artefact and "73.5%
coverage" measured whether any candidate existed rather than whether it was
the ball.

Reproduce with `python run_veo_analysis.py --clip <clip> --out output_veo`.

## Derived parameters

    imgsz          960          upscaled 1.5x from the source
    conf_player    0.08         -> 11.0 players/frame
    conf_ball      0.05         -> 1.07 candidates/frame
    scale          35.2 px/m    a 1.75 m player is 62 px tall
    camera         static       0.28 px/frame, below the 1.0 threshold

The elbow rule picked 0.08 rather than collapsing to the MIN_USABLE_CONF floor
of 0.05, which is the regime it was rebuilt for -- a partial-pitch view never
reaches the roster bound, and the old rule returned the floor for no reason
connected to the video.

## Observed

| | observed | expected | reading |
|---|---|---|---|
| ball coverage | 72.7% | — | matches 720p broadcast |
| ball speed p95 | **332 km/h** | ~120 | impossible |
| players per frame | 10 (max 16) | 22 on a full pitch | partial view, unverified |
| track ids | 287 -> 173 | ~22 | heavy fragmentation |
| passes | 24.7/min | 8-12 | 2-3x over-produced |
| carries | 19.0/min | — | over-produced |
| distance per 90 | 12.0 / 9.3 km | 10-12 | plausible |
| top speed | median 33, max 40 | 30-36 | plausible, clipped at the 40 limit |

## Two predictions that were wrong

**Ball coverage did not collapse.** The detection sweep showed ball recall
falling to 0.10 at reduced input size, and this footage has a ball about 7.8 px
across, so coverage was expected to be the binding constraint. It is 72.7%,
matching the 720p clips. Upscaling to 960 before detection is doing the work.

**The camera is not panning here.** A Veo camera synthesises its view from a
panoramic sensor, and an earlier 2-minute clip measured 3.19 px/frame, so
compensation was expected to matter. This window measures 0.28 px/frame and
compensation is skipped. Camera motion on this footage varies by passage
rather than being a property of the camera.

## The ball track teleports -- everywhere, not here

A p95 of 332 km/h means the selected ball track jumps between objects. Two
claims were made about it here and both were wrong.

**It is not a Veo problem.** Measured with the same diagnostic, the 720p
broadcast footage is worse:

| | Veo 640x360 | SoccerNet w1 720p |
|---|---|---|
| ball speed p95 | 332 km/h | **649** |
| max | 1368 | **2164** |
| p95 at consecutive frames | 299 | **626** |

**It does not explain the event over-production.** The broadcast windows have
worse ball speeds and pass detection above chance at p 0.001 on all four. A
teleporting ball evidently does not prevent pass detection from working.

The cause is detection, not selection. In 54% of Veo frames and 64% of
broadcast frames there is exactly one ball candidate, so the selector has
nothing to choose between: when that candidate is the wrong object, no
selection rule can repair it. Of the steps that exceed the speed budget,
about 57% arrive at a frame with a single candidate.

Raising the motion penalty confirms it. From MOTION_WEIGHT 4 to 150 the p95
is unchanged on every window -- 649, 811, 682, 288 km/h throughout. A 37-fold
increase in the penalty moves nothing, because the frames that produce the
spikes never offered an alternative.

The sweep did improve events slightly, by a different route: pass mean F1
rises 0.648 to 0.668 across the four labelled windows, on a broad plateau
from weight 10 upward. That is adopted. It is not a fix for the ball track.

## What this run cannot tell us

Nothing here is accuracy. Whether the passes it finds are real passes is
unknown and unknowable without labels for this footage. The SoccerNet results
-- pass detection above chance on four windows across two matches -- were
measured on 720p broadcast and do not transfer by assumption to 640x360.
