"""Place a ball on the pitch using the goal, not the centre circle.

`detect_shots.py` finds shots by putting the ball into pitch metres and
looking for a fast one heading at a goal. Until now the only way to do that
was the centre-circle anchor, and shot recall is **0 of 10** because of it:
the anchor is fitted at midfield, and a camera following play into the box
does not show the centre circle. The landmark that *is* in frame when shots
happen is the goal.

This provides the same service from the goal instead, so the detector's
logic is untouched and only its geometry changes.

## How a frame gets a pose

Two pieces, in order:

1. **Once per clip**, hand-clicked goal corners give the camera's position
   through `goal_pose.bundle`. A broadcast camera is bolted to a gantry, so
   that position holds for every frame of the clip -- the bundle reprojects
   all of a clip's corner frames from one position to under a pixel.
2. **Per frame**, the goal detector returns a box. With position already
   fixed, rotation and focal are the only unknowns -- four numbers, and a
   box is four numbers -- so `goal_pose.pose_from_box` solves it.

The cost of the labelling is therefore per clip, not per frame.

## Goal-frame to pitch-frame

`ground_point` answers in goal-frame metres: X across the goal line from the
left post, Z out onto the pitch. The pitch frame the detector works in puts
a goal at x = 0 and the pitch centre line at y = WIDTH/2, so

    pitch_x = Z
    pitch_y = X + (PITCH_WIDTH - GOAL_WIDTH) / 2

and the calibrated goal is always the "left" one as far as the detector is
concerned.

That is a real limitation and not a rounding of one: only the goal that was
calibrated can be shot at. A window showing play at both ends would need
both ends clicked. The labelled windows here are cut around a shot, so the
goal in question is the one in view.

## What it cannot fix

The ground intersection assumes the ball is on the grass, and a struck ball
is in the air. Taking the shooter's feet avoids that for a known shot
moment, but a *detector* has to work from the ball, because the ball's speed
and heading are what make a shot a shot. So the placed position is right
where the ball is low and increasingly wrong as it rises. The anchor
homography has exactly the same flaw, so this is not a regression -- but it
is a ceiling, and it belongs here rather than in a footnote.
"""

from __future__ import annotations

import numpy as np

from . import pitch_model as pm
from . import shot_geometry as sg
from .goal_pose import GOAL_WIDTH_M, ground_point, pose_from_box

# The detector's measured operating point, chosen against its 124 goal-less
# frames: 0.15 finds 6 of 6 shot moments for 4 false frames in 124, where
# 0.25 finds 5 for 2. An earlier version of this file used 0.35 on the
# reasoning that a box only just a goal is not worth solving a camera from.
# That reasoning was never measured and cost most of the coverage -- goals
# found on 60 of 466 sampled frames, 13%, where at 0.15 the same clip gives
# 27 of 30 grid frames around a shot.
MIN_BOX_CONFIDENCE = 0.15

# Poses are smoothed across this many grid steps either side.
#
# Each grid frame solves its own pose from its own box, so box noise becomes
# pose noise and pose noise becomes metres. Measured on consecutive frames
# around a labelled shot, placed positions jumped 10 to 15 m between one
# frame and the next -- and a shot is found by fitting a velocity over six
# frames, which that destroys.
#
# The camera is the reason this is fixable rather than merely reducible: it
# pans, tilts and zooms smoothly, so its rotation and focal are smooth
# functions of time and a jump between neighbours is noise by construction.
# Rotation vectors are averaged componentwise, which is only valid for small
# differences; four frames apart on a broadcast pan they are small.
SMOOTH_STEPS = 2

# How far a pose may disagree with its neighbours and still be used.
#
# Smoothing helps only where there are neighbours to smooth against. Around
# a shot with the goal densely in view -- 27 poses over 120 frames -- it cut
# the 90th-percentile jump between consecutive placed positions from 10.2 m
# to 6.0 m and the worst from 93.9 m to 9.8 m. Around a shot with the goal
# mostly out of view -- 9 poses over the same span -- it did nothing useful,
# because averaging a handful of badly disagreeing poses spreads the error
# rather than cancelling it.
#
# So a pose has to agree with its neighbours to be used at all, which is the
# same discipline `fit_pitch_anchor` applies to anchors: a landmark that
# cannot reproduce is not a landmark. Disagreement is measured where it
# matters, in metres on the pitch, by placing one reference pixel through
# the pose and through its neighbours' average.
MAX_POSE_DISAGREEMENT_M = 4.0

# A pose must sit in a run of this many detections to be used.
#
# Without it the agreement gate is toothless exactly where it is needed. An
# isolated pose has no neighbours to disagree with, so it passes by default,
# and those are the poses that put consecutive ball positions 66 m apart.
# Requiring a run encodes what makes the geometry trustworthy in the first
# place: the goal steadily in view, not glimpsed once.
MIN_POSE_NEIGHBOURS = 3

# How far past a line a ball may be placed and still be believed. Balls do
# go out; rays that graze the horizon go to Australia.
OFF_PITCH_MARGIN_M = 8.0

# A box fit worse than this is not describing the calibrated goal -- most
# likely the goal at the other end, or a false positive on a hoarding.
MAX_BOX_RESIDUAL_PX = 40.0

# Solve a pose every this many frames and reuse it in between. The camera
# moves smoothly and a pose costs an optimisation, so this is the same
# bargain `detect_shots.ANCHOR_GRID` strikes for anchors.
POSE_GRID = 4


def to_pitch(x: float, z: float):
    """Goal-frame metres to pitch metres, with the calibrated goal at x = 0."""
    return z, x + (pm.PITCH_WIDTH_M - GOAL_WIDTH_M) / 2.0


class GoalPlacer:
    """Ball pixels to pitch metres, through a goal pose per frame."""

    #: Which goal the calibration belongs to, in the detector's frame.
    side = "left"

    def __init__(self, poses: dict, grid: int = POSE_GRID,
                 far_limit_m: float = pm.PITCH_LENGTH_M / 2.0):
        self.poses = poses
        self.grid = grid
        # How far from the calibrated goal a placement may be. Halfway for
        # a goal-derived pose, which extrapolates badly the further it
        # reaches; a midfield pose is measured at halfway and gets the
        # whole pitch.
        self.far_limit_m = far_limit_m

    def __len__(self) -> int:
        return len(self.poses)

    def pose_at(self, frame: int):
        index = int(frame)
        pose = self.poses.get(index)
        if pose is not None:
            return pose
        for offset in range(1, self.grid // 2 + 1):
            for candidate in (self.poses.get(index - offset),
                              self.poses.get(index + offset)):
                if candidate is not None:
                    return candidate
        return None

    def place(self, frame, px, py):
        """Pitch coordinates for a pixel, or None where there is no pose."""
        pose = self.pose_at(frame)
        if pose is None:
            return None
        hit = ground_point(pose, float(px), float(py))
        if hit is None:
            return None
        x, y = to_pitch(hit[0], hit[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        # A grazing ray is the failure to guard against: near the horizon a
        # few pixels of ball position become tens of metres of ground, and
        # the result looks like a measurement. The first version of these
        # bounds allowed y from -68 to +136 m -- three pitch widths -- which
        # refused nothing at all.
        #
        # A ball genuinely leaves the pitch, so the margin is a touchline's
        # worth rather than nothing, and the far limit is the halfway line:
        # this placer knows where one goal is, and `MAX_SHOT_DISTANCE_M` is
        # 35 m, so a point in the far half is not a shot at this goal
        # whatever else it is.
        if not (-OFF_PITCH_MARGIN_M <= x <= self.far_limit_m):
            return None
        if not (-OFF_PITCH_MARGIN_M <= y
                <= pm.PITCH_WIDTH_M + OFF_PITCH_MARGIN_M):
            return None
        return np.array([x, y])


def smooth(poses: dict, grid: int, steps: int = SMOOTH_STEPS,
           min_neighbours: int = None) -> dict:
    """Average each pose with its neighbours in time.

    Only rotation and focal are averaged; the camera position is shared and
    exact already. Frames with no neighbour within reach keep their own
    pose rather than being dropped -- a lone pose is noisier, not wrong.
    """
    import cv2

    if steps <= 0 or len(poses) < 2:
        return poses
    order = sorted(poses)
    out = {}
    for i, index in enumerate(order):
        near = [poses[order[j]]
                for j in range(max(0, i - steps),
                               min(len(order), i + steps + 1))
                if abs(order[j] - index) <= steps * grid]
        needed = (MIN_POSE_NEIGHBOURS if min_neighbours is None
                  else min_neighbours)
        if len(near) < needed:
            continue
        rvec = np.mean([p.rvec.reshape(3) for p in near], axis=0)
        focal = float(np.mean([p.focal_px for p in near]))
        rot, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        base = poses[index]
        eye = base.camera_position()
        averaged = base.__class__(
            rvec=rvec.reshape(3, 1), tvec=(-rot @ eye).reshape(3, 1),
            focal_px=focal, cx=base.cx, cy=base.cy,
            reprojection_px=base.reprojection_px, n_corners=0)

        # The gate: place one reference pixel -- low and central, where a
        # ball in play usually is -- through this frame's own pose and
        # through its neighbours'. A pose that cannot agree with the frames
        # either side of it is not describing the same camera they are.
        probe = (base.cx, base.cy * 1.5)
        here = ground_point(poses[index], *probe)
        there = ground_point(averaged, *probe)
        if here is None or there is None:
            continue
        if float(np.hypot(here[0] - there[0], here[1] - there[1])) > \
                MAX_POSE_DISAGREEMENT_M:
            continue
        out[index] = averaged
    return out


def build(video_path: str, frames_wanted, camera_position, focal_seed: float,
          width: int, height: int, detector, grid: int = POSE_GRID,
          conf: float = MIN_BOX_CONFIDENCE, verbose: bool = False):
    """Solve a pose on a grid of frames, from detected goal boxes.

    `detector` is anything with ultralytics' `predict` interface, kept as an
    argument rather than loaded here: the weights are trained on footage
    under a non-commercial agreement and cannot live in this repository, so
    the caller supplies them.
    """
    import cv2

    wanted = sorted({int(f) - int(f) % grid for f in frames_wanted})
    if not wanted:
        return GoalPlacer({}, grid)

    cap = cv2.VideoCapture(video_path)
    poses, tried, detected = {}, 0, 0
    for index in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        tried += 1
        result = detector.predict(frame, conf=conf, verbose=False)[0]
        if not len(result.boxes):
            continue
        scores = result.boxes.conf.cpu().numpy()
        box = result.boxes.xyxy.cpu().numpy()[int(np.argmax(scores))]
        detected += 1
        pose = pose_from_box(box, camera_position, focal_seed,
                             width / 2.0, height / 2.0,
                             frame_size=(width, height))
        if pose is None or pose.reprojection_px > MAX_BOX_RESIDUAL_PX:
            continue
        poses[index] = pose
    cap.release()

    poses = smooth(poses, grid)

    if verbose:
        print(f"  [goal] {detected} goals found on {tried} sampled frames, "
              f"{len(poses)} gave a pose")
    return GoalPlacer(poses, grid)


# --- combining the two landmarks -------------------------------------------
#
# The goal and the centre circle are complementary by construction. The
# anchor is fitted from the centre circle, which is in view at midfield and
# gone once play reaches the box; the goal is in view in the attacking third
# and gone at midfield. Each covers where the other fails, and on
# stoke_1302 the goal alone reaches only 10% of ball frames.
#
# What stops this being free is the frame. The anchor has its own pitch
# coordinates with goals at x = 0 and x = 105, and nothing says its x = 0 is
# the goal the corners calibrated. If it is the other one, the two differ by
# a half turn, and combining them without checking would be worse than
# either alone.
#
# So it is checked rather than assumed: on frames where both can place the
# ball, the disagreement is measured as it stands and again with the anchor
# turned through 180 degrees. The better wins, and if neither agrees the
# two are not combined at all.
#
# ## Measured: this footage cannot decide it
#
# It refuses, and the reason is worth stating so nobody rebuilds this
# expecting a different answer.
#
# The landmarks never share a frame -- not one, on reading_2519's 103 goal
# poses and 117 anchor maps, and still none allowing two frames of slack. A
# tight broadcast framing shows the box or the centre circle, never both.
# The nearest anchor frame to a goal frame is 59 frames away on that clip
# and 812 -- thirty-two seconds -- on stoke_1302.
#
# Bridging with players gives, on reading_2519:
#
#     bridge 3.0 s    6 players    as-is 29.3 m   turned 49.5 m
#     bridge 6.0 s   25 players    as-is 40.0 m   turned 51.2 m
#
# As-is wins both times, by 20.2 m and 11.2 m, under the 30 m margin
# required. So the answer is "probably not turned, but not established".
#
# The residual is not evidence against the anchor, and should not be read
# that way. Over a three-second bridge a player covers twenty metres and
# over six he covers fifty, so a 29-40 m disagreement is exactly what
# player motion alone would produce. The measurement cannot separate a
# wrong anchor from too long a bridge, and the bridge cannot be shortened
# because the landmarks are never close in time here.
#
# What would settle it is footage where both landmarks are in frame within a
# second of each other -- a wider angle, or a full match in which such
# moments occur. Not more code.

# How far apart the two may be, in metres, on frames where both place the
# ball, before combining them is refused.
MAX_LANDMARK_DISAGREEMENT_M = 12.0

# Fewer overlapping comparisons than this and it decides nothing.
MIN_OVERLAP_FRAMES = 15

# Where to compare the two landmarks, as fractions of the frame.
#
# The first version of this compared them at the *ball*, and got zero
# overlapping frames on both clips tried -- the ball is detected on a
# fraction of frames, and requiring it on a frame both landmarks also cover
# made the comparison almost impossible. Any point on the grass does the
# job, and every frame has plenty. These sit low and central, where the
# pitch is and where neither mapping is extrapolating near its horizon.
PROBE_POINTS = ((0.35, 0.72), (0.50, 0.78), (0.65, 0.72))

# How far apart two frames may be and still be bridged by a player.
#
# Measured, the two landmarks never share a frame at all: on reading_2519 the
# goal has poses on 103 frames and the anchor maps on 117, both spanning the
# whole clip, and the overlap is zero -- zero exactly, and still zero
# allowing two frames of slack. That is not a sampling artefact. A tight
# broadcast framing shows either the box or the centre circle, never both,
# so no probe point on any shared frame can decide the orientation, because
# there are no shared frames.
#
# Players can bridge the gap. They are detected on nearly every frame and
# carry persistent track ids, so the same player can be placed through the
# goal on one frame and through the anchor on a nearby one. Twelve frames is
# half a second, in which a sprinting player moves about 4 m -- far less
# than the pitch length that separates a right orientation from a wrong one.
# Twelve frames was the first value and it bridged nothing. Measured, the
# nearest anchor frame to a goal frame is 59 frames away on reading_2519 and
# 812 -- thirty-two seconds -- on stoke_1302. The two landmarks do not
# interleave finely; they occupy different stretches of the clip.
#
# A wider bridge is still decisive because the two hypotheses are not close:
# they differ by a pitch length, about 105 m, while a player covers at most
# 15 m in the two and a half seconds this allows. It cannot rescue a
# thirty-second gap, and is not asked to -- `MIN_ORIENTATION_MARGIN_M`
# refuses rather than guesses there.
BRIDGE_FRAMES = 75

# How much better the winning orientation must be, in metres.
#
# Over a wide bridge the players really have moved, so the two hypotheses
# are compared with a margin rather than a bare inequality. Without one, a
# clip where play happened to travel half the pitch between the two
# stretches could score 60 m against 45 m and pick the wrong answer
# confidently. A wrong orientation puts every anchor-placed ball at the far
# end, which is worse than not using the anchor at all.
MIN_ORIENTATION_MARGIN_M = 30.0

# Fewer bridged players than this and the orientation is a guess.
MIN_BRIDGE_PLAYERS = 20


def turn(point):
    """A pitch position seen from the other end."""
    return np.array([pm.PITCH_LENGTH_M - point[0],
                     pm.PITCH_WIDTH_M - point[1]])


class CombinedPlacer:
    """The goal where it is visible, the anchor elsewhere.

    The goal is preferred wherever it has a pose: it is the landmark that is
    in frame when shots happen, and the one whose geometry was checked
    against a quantity the laws of the game fix. The anchor fills in the
    rest of the pitch.
    """

    def __init__(self, goal: "GoalPlacer", anchor, flip: bool = False):
        self.goal = goal
        self.anchor = anchor
        self.flip = flip
        self.side = goal.side
        self.used = {"goal": 0, "anchor": 0}

    def __len__(self):
        return len(self.goal) + len(getattr(self.anchor, "maps", ()) or ())

    def place(self, frame, px, py):
        point = self.goal.place(frame, px, py)
        if point is not None:
            self.used["goal"] += 1
            return point
        point = self.anchor.place(frame, px, py)
        if point is None:
            return None
        self.used["anchor"] += 1
        return turn(point) if self.flip else np.asarray(point, dtype=float)


def agreement(goal: "GoalPlacer", anchor, frames, width: int, height: int):
    """How far apart the two landmarks put the same pitch point, both ways.

    Returns (median as-is, median turned, number of comparisons).
    """
    straight, turned = [], []
    for frame in frames:
        for u, v in PROBE_POINTS:
            a = goal.place(frame, u * width, v * height)
            b = anchor.place(frame, u * width, v * height)
            if a is None or b is None:
                continue
            b = np.asarray(b, dtype=float)
            straight.append(float(np.linalg.norm(a - b)))
            turned.append(float(np.linalg.norm(a - turn(b))))
    if not straight:
        return None, None, 0
    return (float(np.median(straight)), float(np.median(turned)),
            len(straight))


def agreement_via_players(goal: "GoalPlacer", anchor, players,
                          max_bridge: int = BRIDGE_FRAMES):
    """Compare the landmarks through a player seen on two nearby frames.

    The same track id, placed through the goal on its frame and through the
    anchor on the nearest frame the anchor covers. Returns
    (median as-is, median turned, comparisons).
    """
    anchor_frames = np.array(sorted(getattr(anchor, "maps", {}) or {}))
    goal_frames = sorted(goal.poses)
    if anchor_frames.size == 0 or not goal_frames:
        return None, None, 0

    by_frame = {f: g.set_index("track_id")
                for f, g in players.groupby("frame")}
    straight, turned = [], []
    for frame in goal_frames:
        near = int(anchor_frames[np.argmin(np.abs(anchor_frames - frame))])
        if abs(near - frame) > max_bridge:
            continue
        here, there = by_frame.get(frame), by_frame.get(near)
        if here is None or there is None:
            continue
        for tid in here.index.intersection(there.index):
            a = goal.place(frame, float(here.loc[tid].px),
                           float(here.loc[tid].py))
            b = anchor.place(near, float(there.loc[tid].px),
                             float(there.loc[tid].py))
            if a is None or b is None:
                continue
            b = np.asarray(b, dtype=float)
            straight.append(float(np.linalg.norm(a - b)))
            turned.append(float(np.linalg.norm(a - turn(b))))
    if not straight:
        return None, None, 0
    return (float(np.median(straight)), float(np.median(turned)),
            len(straight))


def combine(goal: "GoalPlacer", anchor, frames, width: int, height: int,
            players=None, verbose: bool = True):
    """Both landmarks in one frame, or the goal alone if they disagree."""
    straight, turned, overlap = agreement(goal, anchor, frames, width, height)
    how = "on shared frames"
    if overlap < MIN_OVERLAP_FRAMES and players is not None:
        if verbose:
            print(f"  [both] {overlap} shared-frame comparisons; bridging "
                  f"with players instead")
        straight, turned, overlap = agreement_via_players(goal, anchor,
                                                          players)
        how = "bridged by players"
        if overlap < MIN_BRIDGE_PLAYERS:
            if verbose:
                print(f"  [both] {overlap} bridged players is too few to "
                      f"decide the orientation; using the goal alone")
            return goal
    elif overlap < MIN_OVERLAP_FRAMES:
        if verbose:
            print(f"  [both] {overlap} overlapping frames is too few to "
                  f"check the frames agree; using the goal alone")
        return goal
    flip = turned < straight
    best = min(straight, turned)
    margin = abs(straight - turned)
    if margin < MIN_ORIENTATION_MARGIN_M:
        if verbose:
            print(f"  [both] the two orientations are only {margin:.1f} m "
                  f"apart, too close to call; using the goal alone")
        return goal
    if verbose:
        print(f"  [both] anchor vs goal, {overlap} comparisons {how}: "
              f"{straight:.1f} m as-is, {turned:.1f} m turned around")
    if best > MAX_LANDMARK_DISAGREEMENT_M:
        if verbose:
            print(f"  [both] they disagree by {best:.1f} m either way; "
                  f"not combining, using the goal alone")
        return goal
    if verbose:
        print(f"  [both] combining, anchor {'turned' if flip else 'as-is'}")
    return CombinedPlacer(goal, anchor, flip)



# --- midfield, from the centre circle ---------------------------------------
#
# The goal covers the attacking third. Midfield comes from the centre circle,
# refitted with the camera's position held (`src/midfield_pose.py`).
#
# Held where, matters more than anything else here. The goal corners alone
# fix the camera only to a line, and on the one-frame clips the point chosen
# on it was 25 m off; with that position 2 of 73 circles fitted on
# stoke_7001 and 0 of 43 on reading_1155. `camera_position.locate` puts the
# camera where the corners' line and the circle's cross, and the caller
# passes that in.
#
# Both are in the calibrated goal's frame, so combining them needs no
# orientation check. That is what closed the anchor route: the anchor had its
# own pitch axes, and whether its x = 0 was the calibrated goal could not be
# settled on footage where the two landmarks never share a frame.

# A circle fit worse than this, in pixels, is not describing the centre
# circle. The validation clips sat at about 4 px median; the usual intruder
# is the penalty arc, which has the same 9.15 m radius and, fitted as if it
# were at halfway, cannot match the projected curve.
MAX_CIRCLE_RMS_PX = 8.0

# How far the refitted focal may stray from the clip's corner-bundle focal.
# A zoom moves it, so this is loose; a fit outside it has matched the wrong
# circle by trading focal against rotation.
FOCAL_RANGE = (0.5, 2.0)


def build_midfield(video_path: str, frames_wanted, camera_position,
                   focal_seed: float, width: int, height: int,
                   find_circle, halfway_line, skip=(), grid: int = POSE_GRID,
                   circles=None, verbose: bool = False):
    """Midfield poses on a grid of frames, from the centre circle.

    `find_circle` and `halfway_line` are passed in rather than imported:
    they live with the probes that measured them, and this module stays
    importable without them.

    `skip` is frames the goal already covers. The goal is preferred there,
    so fitting the circle on them would be work thrown away.

    `circles` is a `camera_position.scan_circles` result, when the caller
    already has one from locating the camera; the scan is the slow part.
    """
    import cv2

    from .camera_position import scan_circles

    from .goal_pose import GoalPose
    from .midfield_pose import fit

    eye = np.asarray(camera_position, dtype=float)
    cx, cy = width / 2.0, height / 2.0
    skip = set(skip)
    wanted = sorted({int(f) - int(f) % grid for f in frames_wanted} - skip)

    from collections import Counter

    if circles is None:
        circles = scan_circles(video_path, wanted, find_circle, halfway_line)
    poses, found, previous = {}, 0, None
    rejected = Counter()
    for index in wanted:
        if index not in circles:
            previous = None
            continue
        found += 1
        arc, half_px = circles[index]
        if half_px is None:
            # The centre circle is the only circle on the pitch with a line
            # through its middle. The penalty arc -- same 9.15 m radius, and
            # the usual impostor -- has none. Without the halfway line this
            # is not identifiably the centre circle, so it is not used.
            #
            # This gate was added believing that stoke_7001's 71 failed fits
            # of 80 were impostors. They were not: 73 of the 80 had a
            # halfway line, and 66 of those fitted the circle alone at
            # 8 px or better. What failed was the camera position they were
            # fitted with, 25 m from where the camera was
            # (`camera_position`). The gate stays because the reasoning
            # above holds; it is not what explained those failures.
            rejected["no halfway line"] += 1
            previous = None
            continue

        seed_rot, seed_f = ((previous[0], previous[1]) if previous
                            else (None, focal_seed))
        rot, focal, rms = fit(arc, half_px, eye, seed_f, cx, cy,
                              seed_rot=seed_rot)
        if rms > MAX_CIRCLE_RMS_PX:
            rejected["circle fit"] += 1
            previous = None
            continue
        if not (FOCAL_RANGE[0] * focal_seed <= focal
                <= FOCAL_RANGE[1] * focal_seed):
            rejected["focal"] += 1
            previous = None
            continue
        previous = (rot, focal)
        rvec, _ = cv2.Rodrigues(rot)
        poses[index] = GoalPose(rvec=rvec, tvec=(-rot @ eye).reshape(3, 1),
                                focal_px=focal, cx=cx, cy=cy,
                                reprojection_px=rms, n_corners=0)

    fitted = len(poses)
    # A lone circle pose is kept. A goal pose comes from four noisy box
    # numbers, so agreement with its neighbours is the only corroboration
    # it has, and the goal path demands three. A circle pose comes from an
    # over-determined fit to hundreds of arc pixels plus the halfway line,
    # and carries its own measured residual, which the 8 px gate has
    # already judged. Demanding neighbours as well threw away every clean
    # circle on stoke_7001: nine passed the fit, a median 32 frames apart,
    # and the three-in-sixteen-frames rule removed all nine. Neighbours are
    # still averaged and still checked for agreement wherever they exist.
    poses = smooth(poses, grid, min_neighbours=1)
    if verbose:
        detail = ", ".join(f"{k} {v}" for k, v in rejected.items())
        print(f"  [midfield] circle on {found} of {len(wanted)} frames the goal "
              f"does not cover; rejected: {detail or 'none'}; "
              f"{fitted} fitted, {len(poses)} kept after agreement")
    return GoalPlacer(poses, grid,
                      far_limit_m=pm.PITCH_LENGTH_M + OFF_PITCH_MARGIN_M)


class LandmarkPlacer:
    """The goal where it is visible, the centre circle where it is not.

    Both poses are in the calibrated goal's frame, so a point placed by
    either is in the same coordinates with no conversion and no guess about
    which way the pitch runs.
    """

    side = "left"

    def __init__(self, goal: "GoalPlacer", midfield: "GoalPlacer"):
        self.goal = goal
        self.midfield = midfield
        self.used = {"goal": 0, "midfield": 0}

    def __len__(self):
        return len(self.goal) + len(self.midfield)

    @property
    def poses(self):
        merged = dict(self.midfield.poses)
        merged.update(self.goal.poses)
        return merged

    def place(self, frame, px, py):
        point = self.goal.place(frame, px, py)
        if point is not None:
            self.used["goal"] += 1
            return point
        point = self.midfield.place(frame, px, py)
        if point is not None:
            self.used["midfield"] += 1
        return point
