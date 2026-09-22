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
    camera         moves        1.22 px/frame mean, compensated

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
| track ids | 287 -> 189 | ~22 | **the ceiling is field of view, not re-id -- see below** |
| passes | 24.7/min | 8-12 | 2-3x over-produced |
| carries | 19.0/min | — | over-produced |
| distance per 90 | 8.1 / 8.9 km | 10-12 | low; scale is unverified without homography |
| top speed | median 13.2, max 31.8 | 30-36 | plausible at last, but r = 0.28 |
| sprints | 0.00 per tracked min | 0.3-0.7 | too few; see the scale caveat |

## Two predictions that were wrong

**Ball coverage did not collapse.** The detection sweep showed ball recall
falling to 0.10 at reduced input size, and this footage has a ball about 7.8 px
across, so coverage was expected to be the binding constraint. It is 72.7%,
matching the 720p clips. Upscaling to 960 before detection is doing the work.

### Burned-in graphics were being tracked as scenery

The background mask is "anything that is not grass", which kept 90% of this
clip's scoreboard. Optical flow then tracked those corners as though they
were the far stand -- and a feature that never moves reports no motion, so
every sample it appears in drags the camera estimate toward zero.

The giveaway is generic, so nothing is hard-coded to one broadcaster: a
burned-in graphic has **structure that does not move while the view does**.
Real scenery has structure and moves; flat grass moves little in appearance
but has no structure to seed a feature on. `camera_motion.static_graphics_
mask` samples 32 frames across the clip and excludes pixels that are both
low-variance over time and high-gradient in the median frame.

| | masked | median step | x range | validator residual |
|---|---|---|---|---|
| before | — | 2.07 px | 1911 px | 0.25 |
| after | 1.4% | **2.25 px** | **2078 px** | **0.21** |

Both moved as predicted: the estimate rose 9% once the static features
stopped holding it down, and it agrees better with an independent
measurement. `auto_tune` now reads 2.05 px/frame against a true 2.25, where
it read 0.28 before this and the sampling fix below.

Downstream, per-track speed reliability rises from 0.28 to **0.39** -- still
under the 0.5 needed to publish a per-player figure, but no longer nothing.

On the four SoccerNet broadcast clips the mask excludes **0.0%** and every
number is unchanged: pass F1 0.85 at +/-2 s, p 0.000, 221 fragments to 139
tracks, re-derived from scratch. No false positives.

The translucent "EliteScholars" watermark is *not* caught, because a
semi-transparent graphic varies with whatever passes behind it and so fails a
test built on pixels that do not change. Widening the threshold to catch it
was measured and rejected: it buys 0.01 of validator residual for four times
as much of the frame masked.

**The camera is panning here, and the pipeline was told it was not.** A Veo
camera synthesises its view from a panorama, and the virtual camera follows
the ball. `auto_tune` measured 0.28 px/frame, below the 1.0 threshold, so
compensation was skipped. The true figure is 2.07 px/frame at the median and
the view drifts **1911 px in x** over three minutes -- nearly three frame
widths.

The estimator took a *median* over five runs of twelve frames. That motion is
bursty: 32% of frames move under a pixel and the rest swing. Simulated on the
measured motion, that sampling returns under a pixel **14% of the time**, and
it did so here. Ten runs of twenty-five frames, averaged rather than
medianed, returns under a pixel in 0% of draws. Fixed.

## Tracking in ground-fixed (panorama) coordinates

Compensating camera motion *is* panorama-space tracking: `camera_motion.
compensate` subtracts the accumulated displacement, so positions are
expressed on the ground rather than in the moving crop. The capability was
already there; the bug above had it switched off.

The check that it works is not subtle. Player positions across the whole clip
span:

| | extent of the player cloud |
|---|---|
| raw, camera frame | **18.1 x 9.8 m** |
| compensated, ground-fixed | **71.2 x 25.2 m** |

Twenty-two footballers do not spend three minutes inside an 18 by 10 metre
box. They appear to only because the camera follows them. Ground-fixed they
occupy a pitch-shaped band, 71 m of a 105 m pitch. The motion estimate is
also accepted by the independent ORB validator (direct 794 px against
accumulated 598 over nine intervals, residual 0.25 of direct).

What it changes:

| | camera frame | ground-fixed | football |
|---|---|---|---|
| distance per 90 | 12.3 / 10.9 km | 8.1 / 8.9 km | 10-12 |
| top speed, median | 22.6 km/h | **13.2** | — |
| top speed, max | 39.4 | **31.8** | 30-36 |
| speed split-half r | -0.01 | **0.28** | >= 0.5 |
| sprints per minute | 0.46 | 0.00 | 0.3-0.7 |
| median per-track p95 speed | 22.8 km/h | 13.8 | — |

**The old numbers were largely camera pan.** In the moving crop a player
running with the play looks stationary and a stationary player looks like
they are sprinting backwards, so per-track speed was measuring the camera.
The ten "sprints" reported before this fix were swings of the virtual camera.

It does **not** fix continuity: 190 tracks become 189. Panorama space fixes
coordinates, not coverage -- a player outside the crop is not recorded at
all, whatever frame the coordinates are expressed in.

### What it exposes

Distance now reads 8.1 km per 90 and sprints 0.00 per minute, both below what
football produces, and they point the same way: the metre scale may be too
large. It comes from median player height at an assumed 1.75 m, which is one
number for a view with strong perspective. The compensated cloud spanning
25 m across a 68 m pitch is the same signal. Speeds and distances are
therefore uncertain by whatever that scale factor is wrong by, which is the
homography this pipeline still does not have.

So the order of the remaining problems has changed: it is no longer camera
motion, it is the absence of a ground plane.

## The homography: what it would fix, and why it is not there yet

`probe_homography.py` takes four measurements.

**One scale does not describe the frame.** The shipped `px_per_m` comes from
the median player height, which is correct at one depth and wrong everywhere
else. Measured by image row:

| | shipped | by image row | factor |
|---|---|---|---|
| Veo | 35.2 px/m | 16.7 to 51.0 | **3.06x** |
| SoccerNet w1 | 34.5 | 22.8 to 47.2 | 2.07x |
| SoccerNet w2 | 34.9 | 21.8 to 47.6 | 2.18x |

Every distance and speed this pipeline reports is therefore wrong by up to a
factor of two either way, depending on where on the screen the player was.
That is the size of the prize.

**The ground-plane model is real but noisy.** For a camera viewing a plane, a
fixed-height object projects to a pixel height linear in image row, zero at
the horizon, with slope equal to the player's height over the camera's.
Fitting it gives camera heights of 6.4 m on Veo and 23-26 m on broadcast --
a Veo mast and a television gantry, which is a good sign. But R^2 is 0.14 to
0.39: youth players differ in height and a bounding box changes with pose and
occlusion, so the relation holds for the population and not for a detection.

**The focal length cannot be recovered from the motion.** The plane model
leaves one unknown, and without it lateral and depth distances scale
differently. Choosing it to make player speed independent of running
direction is appealing and does not survive contact: the lateral-to-depth
p90 ratio is 1.49 on broadcast and 0.78 on Veo, where perspective alone
requires it to exceed 1 on both. Footballers run along the pitch more than
across it, and that confound is larger than the effect.

**Chaining homographies between frames collapses.** The physical camera is
fixed and only pans and zooms, so every pair of frames is related by an exact
homography, and a mosaic should let markings from different moments
accumulate into one pitch. Chained over 426 frame pairs it degenerates into a
radial smear: pairwise projective error compounds, and without bundle
adjustment or loop closure the chain is worthless by the end. Measured
separately, the virtual camera's zoom moves by more than 2% in 4 of 10
sampled steps, ranging 0.81 to 1.31, so translation-only compensation is also
an approximation -- a good one, but an approximation.

**Markings are visible but thin.** They cover 1.1% to 3.4% of the pitch area
in the frames checked -- a line here, an arc there. One or two segments is
not four correspondences, and deciding *which* line a segment is remains the
ambiguity a homography exists to resolve.

### What would unlock it

Not more effort on this clip. A wider, fixed Veo export -- the same thing
per-player continuity needs -- would put several markings in frame at once
and hold the zoom still, at which point a line-and-circle fit against a pitch
model becomes a normal piece of work rather than a research project. Failing
that, four clicked correspondences on one frame of a non-zooming clip would
do it.

Until then the honest statement is that distances and speeds are
**proportional** measurements, good for comparing players and passages within
a clip, and not metric. The per-90 kilometre figures should carry that
caveat wherever they are shown.

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

## Top speed was a jitter spike; fixed on broadcast, absent here

The table above once called a median top speed of 33 km/h "plausible, clipped
at the 40 limit". That was wrong twice over.

`stats.physical_stats` took each track's **maximum** frame-to-frame speed
after a 5-frame median smooth, discarding anything above 40 km/h. A maximum
over a noisy signal measures the noise, and it did: a track's maximum was
about double its own 95th percentile, and 7-17% of tracks reached the 40 km/h
discard threshold, which is a filter catching tracking error rather than a
limit football approaches.

There is no ground-truth speed, so candidates were scored for **reliability**
instead: a statistic that measures a player agrees between the first half of
their track and the second. Over 320 broadcast tracks
(`sweep_top_speed.py`):

| statistic | split-half r | median | population max | at 40 km/h |
|---|---|---|---|---|
| maximum (was shipped) | 0.50 | 30.7 | 40.0 | 13% |
| **95th percentile** | **0.56** | **15.9** | **37.0** | **0%** |
| 90th percentile | 0.56 | 13.1 | 36.3 | 0% |
| fastest 0.5 s window | 0.43 | 15.8 | 39.9 | 5% |

The rolling window is what sport science reports and it came last, because a
maximum *over windows* is still an extreme-value statistic: one bad frame
corrupts every window containing it. The 95th percentile is now shipped.

### On this footage the metric has no signal at all

The fix moves the Veo numbers from a median of 33 km/h to 22.5, which looks
better and is not. Running the same split-half test per clip:

| footage | split-half r |
|---|---|
| SoccerNet 720p, four windows | 0.45, 0.62, 0.49, 0.58 |
| **Veo 640x360** | **-0.01 over 113 tracks** |

Zero. A track's top speed in its first half predicts nothing about its second
half. At 35 px/m a single pixel of jitter per frame is already 2.5 km/h, and
a 640x360 detection jitters by several, so the tail of the speed distribution
is entirely detection error whatever percentile is taken.

`run_veo_analysis.py` now prints this figure with every run and says plainly
when the number should not be reported. The metric is fixed for broadcast and
**should not be shown to anyone for Veo footage** until the underlying
positions are steadier -- which is a detection and smoothing problem, not a
statistics one.

Distance survives the same scrutiny, because it sums displacements rather
than taking an extreme: 12.3 km per 90 across all tracks and 10.8 for those
followed at least 15 s, against the 10-12 football expects.

### Sprints had the same fault and three more

`n_sprints` counted *frames* whose frame-to-frame speed exceeded 20 km/h. At
25 fps one second of sprinting scored 25, the threshold was applied to the
same noisy per-frame signal, and nothing was normalised by how long the track
was followed. It reported a mean of **70 per minute** where football produces
0.3 to 0.7.

A sprint is now a contiguous run above the threshold lasting at least a
second, with speed measured across a 0.4 s centred span. Both constants were
swept (`sweep_sprints.py`), as rate per minute:

| hold | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|
| per-frame speed | 37.85 | 1.34 | 0.43 | 0.21 | 0.13 | 0.13 |
| 0.4 s span | 4.82 | 1.88 | 1.19 | 0.97 | 0.80 | **0.71** |

The per-frame row collapses to nothing: almost every stretch it finds above
the threshold is a blip one or two frames long, which is the signature of
noise. The 0.4 s row falls and then plateaus, which is what a population of
genuine sustained runs looks like. One second is also the conventional
definition in sport science.

Counts per window fall from 1594, 2259, 1143 and 556 *frames* to 16, 22, 7
and 3 *sprints*, at 0.36-1.04 per tracked minute. On this Veo clip: 10
sprints, 0.45 per minute -- in range, and notably more robust than top speed
was, because a second of sustained running is not something jitter produces.

**The per-player count is still not reliable.** Split-half agreement on
tracks of ten seconds or more is 0.26 on broadcast and -0.04 here. The
aggregate rate is worth reporting; attributing sprints to individuals is not.
`run_veo_analysis.py` prints both figures and says so.

## Tracking continuity: the matcher is not the bottleneck

287 fragments become 190 tracks for about 22 players, and per-player speed
and sprint figures fail their reliability tests partly because a player's
minutes are scattered across several ids. Re-identification was the obvious
suspect.

It was measured rather than assumed. `bench_track_continuity.py` cuts real
tracks into pieces with a known gap and asks the matcher to put them back, so
which pieces belong together is known exactly. Counting tracks against a
roster of 22 would have rewarded merging everything; this does not.

**The first version of that benchmark was wrong, and production caught it.**
Cutting a track cleanly leaves the second piece continuing the first's motion
exactly, so the prediction lands almost on it -- median drift 1.6-2.9 m/s.
Real ByteTrack fragments drift 7.8-13.3 m/s, because an identity is dropped
*when the player is occluded*, and the frames the velocity is read from are
the corrupted ones. Tuned on clean cuts, the benchmark recommended a radius
three times tighter, scored better for it, and merged *fewer* fragments in
production. The benchmark now injects noise at each cut, calibrated to
reproduce the observed drift.

On the corrected instrument the slack term rises from 1.5 m to 3.0 m
(held-out F1 0.61 to 0.78) and the drift term stays at 8.0, which turns out
to match the measured real drift -- the right value for a different reason
than it was chosen for. In production that is worth 143, 126, 133 and 98
tracks becoming 139, 120, 130 and 95, with pass and carry detection unchanged
to the second decimal.

**Three to five percent, and per-player reliability does not move at all**
(speed 0.44-0.58 against 0.45-0.62 before). The matcher was not the
constraint.

### What is

Where fragments end says why:

| | ends near the frame edge | starts near the edge |
|---|---|---|
| SoccerNet 720p, four windows | 36-40% | 26-42% |
| **Veo 640x360** | **52%** | 38% |

More than half of all track breaks on this footage are a player crossing the
edge of the view. Nothing a gap matcher does can rejoin those: a player who
leaves and returns twenty seconds later is not a 2.5 s gap, and treating them
as one would be a guess. Only 6-12% of fragments end because the clip does.

So per-player continuity is bounded by how much of the pitch is in frame and
by detections that fail for longer than a couple of seconds -- a detection
and field-of-view problem. Aggregate figures like distance and the sprint
rate survive it, because they pool across fragments. Per-player attribution
does not.

## What this run cannot tell us

Nothing here is accuracy. Whether the passes it finds are real passes is
unknown and unknowable without labels for this footage. The SoccerNet results
-- pass detection above chance on four windows across two matches -- were
measured on 720p broadcast and do not transfer by assumption to 640x360.
