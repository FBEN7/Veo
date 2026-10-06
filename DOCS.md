# Where things are

The repository has grown by measurement: a claim is made, checked against
something it could not have been fitted to, and kept or corrected in the
open. This page says which documents describe the system as it is and which
are the history that got it there.

## Current

| document | what it holds |
|---|---|
| `README.md` | what the pipeline does, how to run it, and the measured state (French) |
| `EVENT_ACCURACY.md` | every event and geometry measurement; opens with a current-state summary |
| `DETECTION_ACCURACY.md` | player and ball detection against human-labelled images |
| `TEAM_ASSIGNMENT.md` | team assignment, and why an earlier figure measured something else |
| `PLAYER_IDENTITY.md` | who did each event: shirt numbers, anonymous identities, measured against clicks and SoccerNet game-state clips |
| `VEO_FOOTAGE.md` | the 640x360 web clip, which is not a Veo camera, and what can be checked without labels |
| `SPEC_rapport_pdf.md` | the specification for the PDF report (French) |
| `DASHBOARD_SETUP.md` | running the dashboard |

## Historical

Written before accuracy was measured against labels. Each carries a banner;
their claims are not the current state.

`DEPLOYMENT_READY.md`, `DEPLOYMENT_STATUS.md`, `OPTIMIZATION_NOTES.md`,
`PIXEL_COORDINATE_FIX.md`, `fix_event_detection_coordinates.md`,
`CODE_REVIEW.md`.

## Tests

`python run_tests.py --list` prints every check and what it covers. The
controls run in seconds without video or model weights; none of them
measures accuracy on football, which needs footage that cannot live here.

## The geometry, module by module

| module | role |
|---|---|
| `src/goal_pose.py` | camera pose from four goal corners, or from a detected goal box once the camera is located |
| `src/camera_position.py` | where the camera is: the goal corners' line crossed with the centre circle's |
| `src/midfield_pose.py` | pose from the centre circle and halfway line, camera position held |
| `src/goal_placer.py` | ball pixels to pitch metres, from the goal where it is in view and the centre circle where it is not |
| `src/ball_path.py` | the ball's most plausible path through each frame's candidates |
| `src/ball_height.py` | flight or roll, from the shape of the ball's path under gravity |

## Probes

Each `probe_*.py` answers one question with a measurement, and its
docstring records the answer. They are kept so a negative result is not
retried blind.

| probe | question |
|---|---|
| `probe_anchor_bias.py` | is what is left of the anchor's error noise, or bias? |
| `probe_anchor_chain.py` | carrying an anchor step by step instead of in one jump |
| `probe_anchor_height.py` | the centre-circle anchor against two independent camera heights -- it fails, 62-72 m |
| `probe_anchor_reproducibility.py` | asking an anchor to be found twice, and what it costs |
| `probe_arc_accumulation.py` | telling the centre circle from the penalty D over several frames |
| `probe_borrow_reach.py` | how far an anchor may be carried, and what each distance buys |
| `probe_centre_circle.py` | finding the centre circle |
| `probe_constrained_anchor.py` | the circle refitted with the camera's position held; player heights as the check |
| `probe_goal_frame.py` | finding the goal frame by hand-written morphology -- it fails |
| `probe_goal_ruler.py` | can a goal box and the ground plane give a distance? width fails, height holds |
| `probe_grass_crossing.py` | the ball leaving the pitch without a pitch map -- it does not work |
| `probe_homography.py` | what a homography would need here |
| `probe_penalty_box.py` | are the penalty-area markings visible in the frames that matter? |
| `probe_pitch_lines.py` | how much pitch geometry is visible |
| `probe_rotation_carry.py` | carrying a goal pose into midfield by rotation -- cuts break it |
| `probe_stoppage.py` | play stopping, read from the players -- it does not work |
| `probe_truncated_box.py` | goal-box poses against corner poses, before and after reading boxes correctly |
| `probe_vanishing_points.py` | rectifying the pitch from the horizon |
| `probe_veo_panorama.py` | is the camera a window over a fixed panorama? |

## Data

SoccerNet videos, labels and anything derived from them -- detector weights,
labelling pages, run outputs -- are under a non-commercial agreement and are
not in this repository. Generators are committed; what they make is not.
