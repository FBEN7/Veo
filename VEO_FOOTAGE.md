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
    camera         moves        2.05 px/frame mean, compensated

The elbow rule picked 0.08 rather than collapsing to the MIN_USABLE_CONF floor
of 0.05, which is the regime it was rebuilt for -- a partial-pitch view never
reaches the roster bound, and the old rule returned the floor for no reason
connected to the video.

## Observed

| | observed | expected | reading |
|---|---|---|---|
| ball coverage | 72.7% | — | matches 720p broadcast |
| ball speed p95 | **329 km/h** | ~120 | impossible |
| players per frame | 10 (max 16) | 22 on a full pitch | partial view, unverified |
| track ids | 287 -> 192 | ~22 | **the ceiling is field of view, not re-id -- see below** |
| passes | 23.3/min | 8-12 | 2-3x over-produced |
| carries | 22.3/min | — | over-produced |
| distance per 90 | 8.0 / 8.9 km | 10-12 | low; scale is unverified without homography |
| top speed | median 13.1, max 30.9 | 30-36 | plausible at last, but r = 0.39 |
| sprints | 0.01 per tracked min | 0.3-0.7 | too few; see the scale caveat |

Re-derived after the graphics-mask fix below, which changed the camera
estimate and therefore every ground-fixed position. Figures elsewhere in this
document that predate it are marked where they differ.

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

**The focal length cannot be recovered from the players' motion.** The plane
model leaves one unknown, and without it lateral and depth distances scale
differently. Choosing it to make player speed independent of running
direction is appealing and does not survive contact: the lateral-to-depth
p90 ratio is 1.49 on broadcast and 0.78 on Veo, where perspective alone
requires it to exceed 1 on both. Footballers run along the pitch more than
across it, and that confound is larger than the effect.

It can be recovered from the *camera's* motion, which is a different thing
and took a while to notice. See "The focal length, and what it was worth"
below.

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

That last paragraph was written before the focal length was measured, and it
is now too pessimistic for broadcast and exactly right for Veo. What
changed is below.

## The focal length, and what it was worth

The missing number turned out to be obtainable, and obtaining it turned out
not to matter nearly as much as this document assumed.

**Where it comes from.** A camera that rotates about its own centre and does
not travel relates any two frames by `H = K R K^-1`. Requiring `R` to be a
rotation pins `f` from `H` alone -- the classic constraint that the first and
third columns of `K^-1 H K` are orthogonal. A broadcast camera on a gantry
and a Veo virtual camera panning across its panorama both qualify. The
homographies come from ORB matches on the same background mask the motion
estimator uses, graphics excluded.

The first implementation returned zero on every clip. That was not the data:
handed a frame warped by a rotation of *known* focal length it also returned
zero, because both constraints had numerator and denominator inverted.
Corrected, it recovers a known f to within a few percent whenever the pan
exceeds about two degrees. The equal-norm constraint is an order of
magnitude noisier and is not used.

On real footage, measured at four baselines from 15 to 120 frames:

| clip | 15f | 30f | 60f | 120f | spread | implied FOV |
|---|---|---|---|---|---|---|
| SoccerNet w1 | 1673 | 1685 | 1629 | 1766 | 0.18 | 42 deg |
| SoccerNet w2 | 1606 | 1589 | 1690 | 1593 | 0.16 | 43 deg |
| SoccerNet w3 | 1719 | 1776 | 1951 | 1790 | 0.27 | 39 deg |
| reading | 1757 | 1801 | 1789 | 1803 | 0.19 | 39 deg |
| **Veo** | 1434 | 2068 | 1637 | 2082 | **0.74** | refused |

Agreement across an eight-fold range of baselines is the evidence: the
projective part of the homography grows with the rotation while the noise
does not, so a spurious f drifts with the baseline. Broadcast holds. Veo
does not, and is refused rather than used.

With f, the whole ground plane follows. Camera heights come out at 21 to 33 m
on broadcast and 6.6 m on Veo -- a television gantry and a mast -- horizons
sit well above the frame, and players land 31 to 74 m away. Three
independent measurements agreeing on one geometry.

**And it does not help.** Projecting onto that plane was measured end to end
against the single scale, with the possession radius re-tuned separately
under each so the comparison is not just measuring thresholds:

| | pass F1, tuned on w1-w3 | held out on Reading |
|---|---|---|
| single scale | **0.793** | **0.706** |
| ground plane | 0.728 | 0.655 |
| ground plane, ball depth from nearest player | 0.757 | 0.667 |

Split-half speed reliability is unchanged on three windows and collapses
from 0.57 to 0.19 on Reading. Distance-rate reliability is equal or worse on
all four and goes negative on Reading. The one thing that improves is the
plausibility of km per 90 -- which is the weak test this document opens by
warning about.

The reason is not subtle. Depth goes as `1 / (row - horizon)`, so the map
multiplies vertical position noise, and at these resolutions that costs more
than the geometry gains. Two attempts to reduce that noise are recorded in
`src/ground_plane.py`: reading depth from the box centre rather than the box
bottom, which keeps `crop_h` jitter out of position and recovered three of
the four windows; and measuring the horizon per window instead of inferring
it from the camera-motion estimate, which is right in principle -- a row
1068 px off-axis moves 36% further under tilt than the centre does -- and in
practice injects enough jitter of its own to halve w3's reliability.

**What does help is the number, not the map.** A scale taken from player
height is the scale for motion *across* the view. Motion *into* it covers
fewer pixels per metre, by roughly (rows below horizon) / f. So one number
from heights overstates pixels per metre, and every distance divided by it
comes out short -- by 9%, 9%, 17% and 26% on the four windows, always in the
same direction. Correcting a single scalar is free where replacing the map
is not, because a uniform rescale cannot move a correlation.

| clip | pass F1 | carry F1 | km/90 | speed r | sprints/min |
|---|---|---|---|---|---|
| w1 | 0.849 -> **0.886** | 0.716 -> 0.687 | 9.6 -> **10.5** | 0.44 -> 0.45 | 0.54 -> 0.58 |
| w2 | 0.698 -> 0.698 | 0.692 -> **0.731** | 11.4 -> 12.5 | 0.58 -> 0.57 | 1.07 -> 1.55 |
| w3 | 0.781 -> 0.767 | 0.714 -> 0.702 | 10.6 -> 12.2 | 0.48 -> 0.47 | 0.39 -> 0.77 |
| reading | 0.727 -> 0.679 | 0.571 -> **0.625** | 8.5 -> **10.5** | 0.57 -> 0.51 | 0.36 -> 0.47 |

Every measured instrument is flat: pass F1 -0.007, carry F1 +0.013,
split-half speed -0.02, averaged over four windows. What moves is the level,
and the level can only be judged by plausibility -- where **the two available
anchors disagree**. Distance per 90 improves on all four, from 8.5-11.4 into
10.5-12.5 against football's 10-12. The sprint rate gets worse, from
0.36-1.07 to 0.47-1.55 against football's 0.3-0.7, because the 20 km/h
threshold is absolute and every speed just rose by a sixth.

It is shipped anyway, on the grounds that the bias is established by
geometry rather than fitted to a target, and knowingly keeping a scale
proven 9-26% too large is worse than correcting it and saying what the
correction does not settle. The correction inherits the focal length's own
uncertainty, so call it 10-25% with a few points either way.

**None of this reaches the Veo clip.** Its focal length is refused, so it
keeps the uncalibrated scale and the "proportional, not metric" caveat above
applies to it in full. `run_veo_analysis.py` prints which of the two it got.

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
and 3 *sprints*, at 0.36-1.04 per tracked minute.

On this Veo clip the figure is **1 sprint over 26.6 tracked minutes, 0.014
per minute** -- far below the 0.3-0.7 football produces. An earlier version
of this section reported 10 sprints at 0.45 per minute and called it in
range. That measurement predates ground-fixed tracking and does not survive
it: the "sprints" were swings of the virtual camera, the same artefact the
compensation table above records as 0.46 per minute becoming 0.00. The
corrected number is the one here.

So sprint counting is more robust than top speed on broadcast -- a second of
sustained running is not something jitter produces -- and on this footage it
now under-reports rather than over-reports. That is consistent with the scale
being too large: a metre that is too big makes every speed too slow, and the
20 km/h threshold is then reached by almost nobody. It points at the same
missing homography as the distance figure does, not at the sprint definition.

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

### The broadcast tuning transfers to this footage unchanged

All of that was measured on 720p and 1080p broadcast, and the radius is in
metres -- so at 640x360, where a player is 62 px rather than 60 and the
metre scale comes from a view with three times the perspective spread, there
was no reason to assume it carried over. The benchmark manufactures its own
truth, so it needs trajectories rather than labels and can simply be pointed
at the Veo clip. It now is.

| gap | broadcast P/R/F1 | Veo P/R/F1 |
|---|---|---|
| 0.2 s | 0.91 / 0.87 / **0.89** | 0.86 / 0.87 / **0.86** |
| 0.4 s | 0.71 / 0.69 / 0.70 | 0.62 / 0.52 / 0.57 |
| 0.8 s | 0.55 / 0.47 / 0.51 | 0.58 / 0.43 / 0.49 |
| 1.5 s | 0.46 / 0.34 / 0.39 | 0.45 / 0.29 / 0.35 |
| 2.5 s | 0.35 / 0.14 / 0.20 | 0.35 / 0.18 / 0.24 |

Re-identification is no worse at a third of the resolution. Swept as a third
held-out set, the Veo column tracks the tuning windows at every one of the 30
settings tried, and the optimum sits in the same place. **Nothing needs
tuning per footage**, and the low per-player reliability on this clip is not
the matcher mis-sized for it.

The prediction drift was checked the same way: 13.4 m/s at the median on Veo
against 15.5 to 17.4 on the four broadcast windows, measured on the fragment
boundaries the matcher declined to join, with the nearest candidate standing
in for the true successor. Those are the hard cases by construction, so the
absolute values run high; what matters is that Veo is not an outlier, so the
noise the benchmark injects at each cut is calibrated for it too.

### The grid had been stopping at the answer

Re-running the sweep exposed a fault in how it had been read. The margin was
searched over 0.5, 1.5 and 3.0, and 3.0 won every row -- recorded as "best at
every drift tried", as though that settled it. It does not: an optimum on the
last column of a grid has not been shown to be an optimum, only that the grid
ran out.

Extending the margins finds a real interior peak at drift 6.0, margin 4.5 --
tuning F1 0.76 against 0.74, Reading 0.82 against 0.81, Veo 0.76 against 0.73.

**It was not adopted, because production disagrees.** At 6.0/4.5 the labelled
windows merge to 142, 122, 131 and 97 tracks where the shipped setting gives
139, 120, 130 and 95: the benchmark prefers a setting that merges *less*,
which is precisely the signature that exposed the clean-cut bug above. And
nothing downstream moves either way -- pass F1 identical to three decimals on
all four windows (0.849, 0.698, 0.781, 0.727), the Veo clip unchanged at 192
tracks and split-half speed 0.39. A 0.02 gain on a synthetic instrument that
buys nothing and points the wrong way in production does not move a shipped
constant.

The widened grid did settle what the sweep is really measuring. Two thirds of
real dropouts are 0.2 s or shorter, so the weighted objective is dominated by
the radius at short gaps, where the margin supplies most of it. Every setting
within 0.02 of the top has a 0.2 s radius between 5.0 and 6.8 m. The sweep
identifies the margin; the drift term it barely constrains, and the match
between drift 8.0 and measured real drift is corroboration from outside it.

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


## The Veo clip can be anchored to the pitch after all

This document has said throughout that pitch coordinates are out of reach on
this footage, and that distances are proportional rather than metric. That is
now wrong, and two mistakes of mine were holding it up.

**The circle detector was deleting the circle.** Finding the centre circle
begins by painting out every straight line, so a conic fit does not have to
contend with a touchline. But the line detector fits straight *chords* to a
large arc -- a 55-pixel chord of a big circle is nearly straight -- so the
erasing removed the circle itself. The bigger the circle in frame, the more
of it went. This clip is the extreme case: its centre circle spans most of
the frame, and the detector found one in **none** of 24 frames. Without the
erasing it finds eight, at a median residual of 0.66 px over 135 degrees of
arc, and the ellipse traces the real circle when inspected.

It was costing detections everywhere, not only here -- 7 to 14 on w1, 2 to 9
on w3 -- and the ones it left were worse. RANSAC rejects straight lines
perfectly well on its own, which is what it is for.

**The horizon was not worth estimating.** The anchor was built around two
vanishing points giving a horizon, a gate to decide when to trust it, and a
transfer to carry it between frames. On identical frames, the anchor that
skips all of that is better:

| | horizon anchor | circle alone |
|---|---|---|
| SoccerNet w1 | 3.3 m | **1.4 m** |
| SoccerNet w2 | 2.5 m | **2.1 m** |
| SoccerNet w3 | 2.2 m | 2.3 m |
| reading | 2.1 m | **0.8 m** |

Not because perspective is absent, but because these are narrow views -- a
broadcast camera following play, and a Veo virtual camera more so. Across a
narrow field of view the projective part of the mapping is small, while a
horizon estimated to remove it carries real error and the transfer adds more.
Removing a small distortion with a noisy correction is worse than leaving it.

That matters here because **this clip has no horizon to estimate**. Four
frames of sixty yield two vanishing points at all, and their spread is 522 to
875 px at every gate tried. Veo shows one family of markings at a time. An
anchor needing only the circle is the only kind this footage can support.

### Where it lands

| | anchored frames | within reach of one |
|---|---|---|
| SoccerNet w1 | 62% | 97% |
| SoccerNet w2 | 35% | 97% |
| SoccerNet w3 | 40% | 85% |
| reading | 35% | 83% |
| **Veo** | **30%** | **68%** |

Frames without their own anchor borrow the nearest, as `H_a W^-1` with W
matched between the two frames -- no focal length involved, which matters
given how much of this project that quantity has cost. Accuracy holds out to
eight seconds: median 1.6 to 2.2 m across all clips, against a measured
chance floor of 5.1 to 5.6 m. What falls with distance is the odds of a
borrow succeeding, from 97% at a third of a second to 40% at eight.

**Veo's margin is the narrowest.** Measured on its own, the clip anchors 9 of
10 circle frames at a median 3.7 m against a 4.9 m chance floor -- better
than nothing by a quarter, where broadcast beats its floor by four fifths.
Pitch coordinates exist on this footage now; they are not yet good enough to
put a shot in the right part of the box.

> **That last paragraph was wrong, and the error was in the reporting rather
> than the anchor.** `fit_pitch_anchor.py` had been changed so that the
> anchor it *builds* takes the horizon at infinity, but the `main()` that
> prints the figures still forced the vanishing-point route, which this
> document had already measured as worse and which the Veo clip cannot
> support at all. So the headline script was scoring a method that no longer
> ships. Corrected, the same clip reads **1.6 m against a 4.7 m floor** --
> 66% below chance, in line with SoccerNet w3 (2.1 against 5.5) and w2 (1.9
> against 5.2), not far behind them. Veo was never the outlier. The figures
> in the table above, which came from `anchored_frames` directly, were right
> throughout.


## Trying to close the gap

The previous section ended by saying the Veo anchor was not yet good enough
to place a shot within the right part of the box. That turned out to be two
claims, and they came apart under examination.

### How good it actually is

Sampled at 120 frames rather than 40, and repeated with three different
draws, the Veo clip gives 34 to 36 anchors each time and:

    seed   anchors   median    p90   chance floor
      0         36     2.0m   7.3m           6.3m
      1         35     1.6m   6.6m           6.7m
      2         34     2.2m   7.3m           6.3m

**2.1 m against a 6.4 m floor**, pooled over 105 anchored frames -- 67% below
chance, where the broadcast clips sit at around 0.9 m. Veo is worse than
broadcast and not by the margin previously reported. The 3.7 m figure came
from a reporting path that had been left behind: `fit_pitch_anchor.py` built
the circle-only anchor but its `main()` still scored the vanishing-point one,
which this document had already measured as worse and which this footage
cannot support at all.

The spread across draws, 1.6 to 2.2 m, is worth noting on its own. Any single
run of this measurement carries a few tenths of a metre of sampling noise,
and earlier numbers quoted to one decimal place from eight or ten frames were
more precise-looking than they were.

### Is the metric even measuring the anchor?

Before trying to improve anything, a question that had never been asked: what
does `marking_error` read when the anchor is exactly *right*? Take a real
anchor, declare it correct, generate the pitch's true markings, project them
through it, cut them into 55-pixel pieces with realistic endpoint noise, and
score them the same way.

    clip            perfect anchor   real anchor
    SoccerNet w1             0.06m         0.90m
    reading                  0.02m         0.82m
    Veo                      0.07m         2.90m
    pooled                   0.06m         1.10m

Six centimetres. The metric is not the limit; the metre is really the anchor.

### Four ways of improving it, none of which worked

**Refine each anchor against the straight markings.** Fitted on half the
segments, scored on the other half: pooled 1.2 m to 1.3 m. It moves an anchor
by half a degree and 0.8 m and changes nothing.

**Refuse anchors that contradict their neighbours.** Two anchors and a warp
measured from image features must agree about a pitch that did not move, and
that test needs no markings at all -- so it would work at run time, on a
frame showing nothing but grass. It does not work: refused anchors score
1.4 m against 1.0 m for kept, and refusing the same number at random scores
1.1 m.

**Let every anchor in the clip vote.** Blocked by feature matching rather
than by geometry, and on the Veo clip most of all: good matches per hop run
245, 0, 0, 0, 0, 0, 133, 542, 645, 1114 -- either hundreds of features match
or none do, and the runs of zeros are texture-poor grass at 640x360 seen by
a corner detector. Where enough anchors did tie together, twenty voters did
not beat one.

**Correct a per-clip bias.** Since all three failures were failures of
averaging, the error looked like it must be systematic. One correction per
clip, fitted on half the anchored frames and applied to the other half, makes
four clips of five distinctly worse.

### What the failures did turn up

The fits kept driving the scale to the edge of whatever bound they were
given -- 10% of 12% per frame, 20% of 20% per clip, three clips out of five,
always shrinking. The obvious suspect was the metric, and the synthetic
control clears it: shrinking a correct anchor by 10% costs 1.09 m while
growing it by 10% costs 0.40 m, so `marking_error` punishes shrinking nearly
three times as hard. What rewards it is the *fitting* loss over detected
segments. Shrinking drags spurious detections -- a shadow, a boot, the edge
of a technical area -- toward the middle of the pitch, where the model lines
are close together and any stray point lands near one. The fits were
exploiting their own outliers. That leaves the scoring trustworthy and the
fitting not, which is the right way round, since every accuracy figure here
rests on the first.

And a structural fact about this footage, which is the most useful thing to
come out of the attempt. **The Veo camera is fixed.** A pure translation
explains a pair of its frames to 1.0 px at a gap of 25 frames but is out by
8 to 19 px beyond 100; a translation *with a scale* holds at 0.92 to 1.29 px
all the way to 2000 frames, matching a full eight-parameter homography. The
crop slides and zooms; the camera behind it never moves. So this clip has one
pose relative to the pitch, every anchored frame measures the same quantity,
and a frame with no markings in it could still be placed. Collecting that
needs a matcher that works on grass, which ORB does not.

### Where the real risk is

Not the median. The cross-frame consistency check -- two anchors compared to
each other, touching no markings -- finds a quarter to a half of all pairs
disagreeing by more than five metres, with worst cases of 20, 50 and 80 m,
and the Veo anchors have a p90 of 7.2 m. Those frames do not look wrong.
A shot placed a metre out is in roughly the right part of the box; a shot
placed fifty metres out is at the other end of the pitch, and nothing about
it announces itself. Anything built on these coordinates needs to survive
that tail, and none of the four attempts above made a dent in it.


## Fixing the tail

The section above ended by naming the tail as the real risk: a quarter to a
half of anchor pairs disagreeing by more than five metres, worst cases of 20,
50 and 80, and none of four attempts denting it. Those attempts all treated
it as a statistical problem -- find the outliers, average them away, refuse
them. That was the mistake. A fifty-metre error is not a statistical event.
It is a specific error with a specific cause, and asking *which* cause found
two of them straight away.

`diagnose_tail.py` applies each mistake the anchor could be making to each
anchored frame and asks whether the error collapses.

### The penalty D, mistaken for the centre circle

**Twenty-one of the twenty-four frames scoring worse than 5 m come back to
about 2 m when the map is shifted by exactly 41.5 m along the pitch** --
52.5 minus 11, the centre spot against the penalty spot. Those frames had
anchored on the D outside the penalty area instead of the centre circle.

It is an easy mistake and an almost undetectable one. Both arcs are struck at
9.15 m because both come from the same measurement in the laws of the game,
so an ellipse fitted to the D has the right scale, a good shape, and a median
residual of 0.7 px -- identical to the frames that got it right. Everything
about the fit is sound except which arc it is.

Geometry separates them and nothing else does. The D is only the part of its
circle lying outside the penalty area: the penalty spot is 11 m from the goal
line and the area's edge is 16.5, so the chord sits 5.5 m from a 9.15 m
centre and the arc spans `2·acos(5.5/9.15)` = **106 degrees, never more**.
The gate was 120 degrees, set as a noise threshold with no idea it was also
the only thing standing between the centre circle and the D.

    min span   anchors kept   median   p90     over 5 m
        120d        129/129     1.0m   7.3m         19%
        180d         78/129     0.7m   3.3m          8%
        200d         70/129     0.6m   2.1m          4%
        220d         60/129     0.6m   1.5m          3%

200 degrees, where the tail falls three and a half times *and* the median
improves -- the sign that what is being refused was wrong rather than merely
marginal.

Worth recording what did **not** separate them: whether a line passes through
the circle's centre. The halfway line bisects the centre circle; the D's
chord misses its centre by 5.5 m, so this ought to be decisive. It is 67%
against 69%, because `halfway_line` accepts a miss of 22 px, which at these
scales is wide enough for the chord to pass as a diameter.

### Which way round the pitch is, decided by an array index

A circle is unchanged by turning it through 180 degrees, and so is the
halfway line, so between them the orientation is ambiguous.
`metric_from_circle` resolved the remainder with `arctan2` of the halfway
line's direction vector -- built from a detected segment's two endpoints, in
whatever order `HoughLinesP` listed them. Reverse the order and the pitch
turns end for end.

**Nothing caught it because a pitch is exactly symmetric under that
rotation.** 105 minus each line across it gives back the same seven numbers;
68 minus each line along it, the same six. A flipped anchor puts every
marking on a real pitch line, so `marking_error` scores it *identically* --
as would any check built on distance to the markings, which is every accuracy
figure in this project.

It is also the error that matters most for what the pipeline is for. A shot
flipped end for end is attributed to the other goal, and a shot from six
yards becomes a shot from ninety-nine.

What does see it is comparing two frames to each other: one pair in ten was
flipped relative to its partner, and those pairs sat **34.5 m apart against
1.1 m** for the pairs that agreed.

The absolute orientation cannot be recovered from markings -- same symmetry,
there is no telling one end of a bare pitch from the other -- and does not
need to be. A clip only has to agree with itself. The camera does not orbit
the pitch, so requiring that moving *down the image* moves toward increasing
y pins the choice to something physical and frame-independent.
`test_pitch_orientation.py` holds it there.

### What the two fixes bought

| | before | after |
|---|---|---|
| worst disagreement between two anchors | 80.6 m | **7.6 m** |
| pairs disagreeing by over 5 m | 27-50% | 0% on three clips of four |
| median disagreement | 1.3 m | 0.6 m |
| propagated error, a third of a second | 1.6 m | **0.7 m** |
| propagated error, eight seconds | 2.0 m | **1.1 m** |

### And what they cost

| clip | anchored | within reach |
|---|---|---|
| SoccerNet w1 | 62% -> 42% | 97% -> 77% |
| SoccerNet w2 | 35% -> 12% | 97% -> 61% |
| SoccerNet w3 | 40% -> 25% | 85% -> 60% |
| reading | 35% -> 22% | 83% -> 54% |
| **Veo** | 30% -> 10% | **68% -> 25%** |

Coverage roughly halved, and worst on the footage the product is for. Veo now
has too few anchors for two of them to overlap, so the consistency check
cannot see it at all.

The trade is the right one -- an anchor 41.5 m out is worse than no anchor,
because nothing about it looks wrong -- but it is not the best available. The
frames being discarded are not bad frames. They are frames looking at the
penalty area, which is where shots are.


### Trying to get that coverage back, and the trap it walked into

The discarded frames are penalty-area frames, and a D-anchored frame is not
wrong by a random amount -- it is wrong by exactly 41.5 m, in a direction
the arc ought to give away, since the D always bulges away from its goal.
So: identify the D positively, and re-anchor on it deliberately.

Identifying it took two goes. Arc span alone is not enough, because a centre
circle with most of itself hidden is also a short arc, and "correcting" one
of those by 41.5 m creates the very error being removed; measured that way it
improved 54% of the frames it touched, a coin toss. What a D has and a
partial circle does not is its chord -- the penalty-area line that cut it,
lying 5.5 m from the centre. Requiring that took the improvement to 94%, and
the frames came back to 2.0 m against a 3.7 m chance floor, from 5.3 m read
as centre circles.

**Switched on, it was a disaster.** Disagreement between two anchors on the
same clip went from 0.6 m to 2.7 m, the worst case from 7.6 m to 114 m, and
two clips of five ended up with median disagreements of 40 m and 23 m.

The two measurements do not contradict each other, and the reason is the
symmetry again. **`marking_error` cannot tell the left penalty spot from the
right one.** Shifting the map to x = 11 and shifting it to x = 94 put the
markings onto mirror images of one another, and the model is symmetric about
x = 52.5, so the two score identically. The 94% confirmed that the
correction was 41.5 m and said nothing whatever about its sign -- and the
sign is the whole question.

That is the same blindness that had just been found hiding the orientation
flip, written up in the section above, and then walked straight into again.
The rule, stated where it will be read next time:

> A degree of freedom that the pitch's symmetry hides cannot be validated
> against the markings. It can only be checked by comparing frames.

`use_penalty_arc` is off. Getting it right needs evidence from outside the
pitch's own symmetry -- which way the camera has panned across the clip,
which goal play is heading toward, or the centre-circle anchors on
neighbouring frames voting on which end this is. The last is the most
promising and is not attempted here.

It is worth knowing what solving it would be worth, so the run was kept:

| clip | within reach, D off | D on |
|---|---|---|
| SoccerNet w1 | 77% | **100%** |
| SoccerNet w2 | 61% | 95% |
| SoccerNet w3 | 60% | 99% |
| reading | 54% | 69% |
| **Veo** | **25%** | **59%** |

Coverage roughly doubles, and on broadcast it becomes essentially complete.
That is the whole of the cost the span gate imposed, handed back, for the
price of one bit per frame.

And a third instance of the same blindness, in the same table: propagated
error with the D anchors switched on reads 1.5 to 1.9 m, perfectly
respectable, on the very anchors that the frame-to-frame check finds 114 m
apart. Every marking-based number in this project shares that limitation.
They measure how well a map agrees with a symmetric pattern, and there are
two ways to agree with it.


## Direction of play settles the end; it does not settle the arc

The section above left the penalty-D anchors switched off because nothing in
a frame can say which of the two penalty spots its arc is struck from -- the
markings are symmetric, so x = 11 and x = 94 score identically against them.
The way out was to stop asking the pitch and ask the clip.

**The camera follows play.** Its accumulated pan says which half of the
pitch is in view, measured against the frames whose centre circle pins them
to the middle. That is evidence from outside the geometry, and it asks very
little of the measurement: the two candidate spots are 41.5 m apart while
the pan drifts about 4 m over a clip. Ten times the margin, for one bit.

The arc's own bulge gives a second, independent sign -- the D swells away
from its goal -- so the two must agree or the frame is refused.

### It works

Across every version tried, **not one pair of anchors landed near 83 m
apart**, which is what the two penalty spots are and therefore what a wrong
end costs. Before the camera was consulted, the bulge-only version produced
114 m disagreements routinely. The orientation question the D posed is
answered.

### The arc is a different problem, and it is not solved

The D anchors that survive are wrong by 30 to 43 m. 41.5 m is the distance
from the centre spot to a penalty spot, which is the signature of a
half-hidden centre circle being shifted as though it were a D. Three filters
were added, each measured:

| filter | D anchors | pairs involving one |
|---|---|---|
| chord at 5.5 m from the centre | 21 | median 30.4 m |
| + the arc must lie on one side of it | 7 | median 42.9 m |
| + the camera may answer "midfield" | 3 | median 30.5 m |

Each removes more of them and the survivors stay wrong. By the last row the
path contributes three anchors across five clips and still misplaces them,
so it earns nothing and stays off.

The difficulty is that a centre circle seen with most of itself hidden
looks, to every test available in a single frame, exactly like a penalty
arc: same 9.15 m radius, same residual, a line near where a chord should be,
and enough of the visible arc on one side of that line. Telling them apart
needs something the frame does not contain -- most plausibly the
neighbouring frames, where the same arc is seen more fully.

What ships is unchanged and unaffected: circle-circle anchors agree to
**0.7 m**, worst case 7.5 m, nothing past 20 m.

### What this means for which goal is which

Two different questions have been run together in these notes and are worth
separating now that one of them is answered.

*Does the clip agree with itself?* Yes. Every frame uses the same
orientation convention, so a shot at one end is attributed to that end
throughout, and the camera's pan can place a frame on the correct half of
the pitch.

*Which physical goal is it?* Still unknown, and geometry cannot say -- a
bare pitch is symmetric. It also does not matter for xG, which is a
surprise worth stating plainly: under the flip, a point at (11, 34) maps to
(94, 34), whose distance and angle to *its* goal are identical. Shot
distance and goal-mouth angle are invariant. What would need the absolute
answer is attributing a shot to a team, which needs team identity anyway and
therefore has to come from outside the geometry regardless.


## Using the neighbouring frames to tell the arc apart

The previous section ended by saying that a half-hidden centre circle and a
penalty D are indistinguishable within one frame, and that the neighbouring
frames -- where the same arc is seen more fully -- were the most plausible
way out. Two mechanisms were tried.

### Pooling the arc pixels: it makes the separation worse

An arc painted on the grass does not move, so warping a neighbour onto this
frame lands its arc pixels on the same circle, and where the two frames hide
different parts of it the union shows more. The quantity that separates the
two arcs is exactly the one that should grow: the D is 106 degrees of its
circle and no number of frames can make that bigger, while a centre circle
is a whole one.

Over 233 ambiguous frames, 43 of them labelled by a neighbouring anchor:

| | circle (n=32) | penalty D (n=11) | separation |
|---|---|---|---|
| span alone | 180° | 110° | **70°** |
| span pooled | 190° | 150° | **40°** |

Pooling does open the circles out, 180° to 190°. It opens the penalty arcs
further, 110° to 150° -- which is the prediction failing, not just a weak
result, since the D was supposed to be incapable of that. The gap the rule
depends on halves. Three runs at different sample sizes all land below the
majority class: 63% against 84%, 86% against 91%, 67% against 74%.

The obvious excuse is bad warps smearing the arc, and the residuals refuse
it: 0.80 px on the frame's own pixels against 0.88 px pooled, with 7.8 times
as many pixels from 7 neighbours. The alignment is fine and the pixels are
real -- they are just not all arc. Pooling brings in every other marking the
neighbours contain, and a D sitting inside a penalty area has a great deal
of such company where a centre circle at midfield has little. The method
adds most noise exactly where it needed to be cleanest.

### Carrying a neighbour's anchor: it works, and it is no use

Put the arc's centre through a confident neighbour's map, carried across the
measured warp, and see whether it lands at the centre spot or at a penalty
spot 41.5 m away. That is reliable; it is what produced the labels above.

But it needs a confident anchor within warp range, which is exactly the case
where propagation already covers the frame. Using it would swap a borrowed
anchor good to 0.7 m for a locally fitted D anchor good to 2.0 m.

### And the population is smaller than the effort

The labelled penalty arcs are 11 of 43, and 8 of 51 and 2 of 22 on the other
runs. Most ambiguous arcs are partial centre circles, not Ds. Even a perfect
classifier would rescue few frames, which caps what this path can be worth
however it is identified -- and is the argument for stopping rather than
trying a fourth mechanism.

`use_penalty_arc` stays off. What ships is unchanged: circle-circle anchors
agreeing to 0.7 m, worst case 7.5 m, nothing past 20 m.


## A detector for the markings

Six attempts to separate a centre circle from a penalty arc by geometry have
failed, and the last section concluded that the only evidence never tried
was the picture itself. So: train something.

### The labels are already there

Where a frame carries a trusted anchor, the pitch model says where every
marking should appear, so each *detected* marking pixel can be handed the
identity of the model marking it lands on. The anchor supplies identity, the
detector supplies position, and neither is asked for what it is bad at --
which matters, because 0.7 m of anchor error is about 25 px against a 12 cm
marking, so labels drawn by projecting the model would sit well off the real
paint. Identity assigned to pixels that were detected is immune to that,
given the closest pair of markings is 5.5 m apart.

Classes are symmetric on purpose: "goal line", never "left goal line". The
end-for-end symmetry that defeated every attempt to name *ends* becomes,
here, the reason the classes are clean and horizontal flips are free
training data.

**One trap, caught by the output.** Labelling only frames that own an anchor
produced 3439 centre circles, 2457 halfway lines and zero penalty arcs --
the class the whole exercise is for. That is circular: an anchor needs a
centre circle, a centre circle is at midfield, so anchored frames never show
a penalty area. Labelling from *borrowed* anchors as well, which are good to
0.7-1.1 m and reach where the camera went, took penalty arcs from 0 to 217.

### Crop size decides it, and differently for each question

| held out | 9 m crops | 24 m crops |
|---|---|---|
| SoccerNet w1 | 72% | **98%** |
| SoccerNet w3 | 50% | **98%** |
| SoccerNet w2 | 50% | 50% |
| reading | 50% | 50% |
| mean | 55% | **74%** |

Balanced, so chance is exactly 50%, and held out by clip -- crops from one
clip share a pitch, a camera and a colour grade, so testing within a clip
asks whether a picture already seen can be recognised.

A fixed *pixel* crop was itself a defect: at 17 px/m on broadcast the old
160 px covered 9 m, on Veo at 35 px/m under 5, so the same marking was shown
at wildly different scales. Crops now cover a fixed 24 m of pitch, which the
anchor's own Jacobian converts to pixels, so every clip presents a marking at
one physical scale whatever its resolution.

Nine metres does not reach the penalty-area line that cuts an arc 5.5 m from
its centre. Twenty-four metres does, and that is the whole difference.

The same change makes the nine-class problem *worse*, 61% to 50% against a
60% floor -- a 24 m crop centred on a touchline pixel holds a lot that is
not the touchline. Circle-against-arc is a question about surroundings;
"which line is this" is a question about the thing itself. One classifier at
one scale for both was the wrong shape.

### How far to trust it

Four initialisations per fold, every one reported:

    held out   seed 0   seed 1   seed 2   seed 3
    w1            98%      97%     100%      99%
    w3            98%     100%      97%      51%
    w2            52%      53%      84%      50%
    reading       52%      50%      50%      50%

Learnable, and not yet general. On w1 every seed lands near perfect, which
is a real signal rather than a lucky draw; on reading no seed leaves chance.
With 217 penalty arcs in total and about 350 examples a class once balanced,
that is roughly what should be expected -- and it is the first positive
result this problem has produced after six geometric attempts.

**It is not wired into the anchor and should not be until it is stable.**
What would stabilise it is more footage, and for the product that means more
Veo: four of the five clips are SoccerNet, whose terms are non-commercial
and no-redistribution, so nothing learned from them can ship, and this Veo
clip yields zero penalty arcs of its own.


## Wiring the classifier into the anchor

The classifier was left out of the pipeline last time because it is at
chance on two clips of four. Wired in, with three safeguards that turn an
unstable signal into a conservative one: it may **abstain** when no crop is
confident or the vote is close, it **votes** over a dozen crops taken along
the arc rather than trusting one, and it is asked at the **24 m scale it was
trained at**, converted to pixels through the anchor's own Jacobian.

Each clip is judged by the model trained *without* it. A model that had seen
the clip would be marking its own homework, and whether it generalises is
the entire question.

### It is safe, which is new

    gates + classifier   circle-circle              pairs with an arc
    strict               38 pairs, 0.6 m median     1 pair, 4.2 m
                         worst 7.6 m, 0% past 20 m  0% past 20 m

No pair past 20 m. The three previous penalty-arc anchors put 57%, 100% and
70% of their pairs there. That failure mode -- an anchor 41.5 m out, looking
perfectly healthy against the markings because the pitch is symmetric -- is
the one that killed every earlier attempt, and it does not happen here.

### It is also nearly empty

Two arc anchors across four clips. The coverage the span gate cost is not
meaningfully recovered.

The classifier is not the bottleneck; the geometric gates in front of it
are. Relaxing them was the obvious next move and was measured:

| gates | arc anchors | arc pair disagreement |
|---|---|---|
| strict | 2 | **4.2 m** |
| relaxed | 3 | **17.5 m** |

One extra anchor, and the one arc pair that can be checked moves most of the
way to a blunder. Reverted. One pair is not a sample and this settles little
on its own, but it points the same way as the earlier measurement -- the
one-sided test took 21 candidate arcs down to 7 -- and when the failure mode
is a shot at the wrong end of the pitch that nothing downstream would
notice, the tie goes to the configuration that has never produced one.

### Where that leaves it

`use_penalty_arc` stays off by default, and the reason is now specific
rather than cautious. On the footage the product is for there is no usable
model at all: the Veo clip contributes zero penalty arcs to train on, and
the weights that do exist are trained on SoccerNet, whose terms are
non-commercial and no-redistribution, so they are as unshippable as the
frames. `marking_model.load` returns None when no weights are present and
every caller carries on without it, so the anchor with no model behaves
exactly as it did before any of this existed.

What is now in place and measured: a labeller that costs nothing, a
classifier that works where it has seen enough, a wiring that cannot do
damage when it is wrong, and a test that would catch it if it did. What is
missing is Veo footage with penalty areas in it.


## Shots, and the first xG this pipeline has produced

The xG model has been finished and verified since it was fitted -- it
recovers known coefficients to a few percent and signs off with "the
pipeline is ready for real shots" -- and nothing had ever handed it one,
because nothing detected a shot. `run_veo_analysis` still prints *goals,
shots and out-of-play are disabled: no homography*, which stopped being true
when the anchor started working.

A shot, in terms the pipeline now has: a ball over 13 m/s on the ground,
sustained, on a line crossing a goal line inside the mouth, from within
35 m. The ball track supplies the first; the anchor supplies metres for the
rest.

**Which goal does not need to be known.** A shot is aimed at the goal it is
aimed at -- take the one the ball is travelling towards and the distance and
angle are the same numbers either way. The symmetry that cost this project
so much is, for xG, irrelevant. It would matter only for saying which *team*
shot, which needs team identity regardless.

### What can be measured, and what cannot

Recall cannot be, and the reason is exact: the labels hold 50 shots and
goals across two full matches and not one falls inside the video that
exists. The Stoke clip covers 320-410 s with labelled shots at 261 and 803 s
either side; the Reading clip covers 3075-3165 s with shots at 2878 and
3377 s. Six minutes of football, verified shot-free.

Verified shot-free is worth something. **Nothing fires on any of it** -- zero
detections on all five clips -- and a synthetic control rules out the dull
explanation, since a detector that never fired would score the same.

A shot planted into each real clip, through that clip's own anchors, is
found once on every one:

| clip | planted | recovered | speed (planted 22 m/s) | xG (true 0.103) |
|---|---|---|---|---|
| SoccerNet w1 | 16.0 m | 15.8 m | 22 m/s | 0.105 |
| SoccerNet w2 | 16.0 m | 16.1 m | 22 m/s | 0.101 |
| SoccerNet w3 | 16.0 m | 15.8 m | 22 m/s | 0.105 |
| reading | 16.0 m | 16.5 m | 22 m/s | 0.097 |
| **Veo** | 16.0 m | **8.5 m** | 22 m/s | **0.298** |

Broadcast recovers the shot to half a metre and its xG to 0.006. That is the
entire chain working at once.

One fix was needed to get there. Velocity had been a difference between the
ends of a 0.12 s window, so the 0.6 m by which neighbouring anchors disagree
became 5 m/s of speed that was not there -- an injected 22 m/s shot read
back as 34 on one clip and 15 on another. A least-squares line through the
window averages that down instead of amplifying it.

### The Veo row is a timing error, and it runs one way

Not a position error. The shot was planted at frame 292 and found at 300,
and eight frames at 22 m/s is 7.0 m of travel: 16.0 - 7.0 is the 8.5 m
reported. The ball really was 8.5 m out when this first saw it.

It saw it late because only **15% of Veo ball positions have a pitch map** --
514 of 3337. The opening of the flight was invisible, so detection began
part-way through. The consequence is specific and one-directional: a shot
picked up late is reported closer to goal than it was struck, and closer
means higher xG. 0.103 became 0.298, nearly threefold.

So on this footage shots are found and overstated, and the cause is anchor
coverage, not the detector. Broadcast at 38-66% placement does not show it.
The thing that would fix it is the thing that has been the answer for three
sections running: more anchored frames, which means more Veo footage.
