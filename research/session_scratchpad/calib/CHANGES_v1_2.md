# SoccerTrack v2: release v1.2

Release v1.2 fixes the synchronisation of match 132877 (M10) with its video, makes every id
unique across the 20 halves, removes placeholder object records, and gives every ball-action
event its frame number. It covers `gsr/`, `ball/` and `bas/`. The videos, `mot/` and `raw/`
are unchanged.

## Changes from v1.1

**M10 (132877) re-synchronised.** In v1.0 and v1.1 the pitch positions of M10 were about 1 s
out of step with the video in both halves, and its image boxes were not on the players: they
lagged the pitch positions by 25 frames and were typically more than 100 px from the player. In v1.2:

- M10's game state is aligned to its video by its pitch positions, so annotation frame *k* is
  video frame *k* in pitch space, as in the other nine matches. The ball arrays and the
  ball-action events of M10 move with it; M10 now has all 2,278 of its benchmark events, including the 3 that v1.1 lost.
- M10's `bbox_image` is regenerated: each box is placed by projecting the object's own
  `bbox_pitch` through M10's released calibration (`raw/132877/132877_camera_intrinsics.npz`),
  with the feet at the bottom centre of the box, and its size is taken from a smooth
  position-dependent model of person-box size in the same video. These boxes are derived
  from the pitch positions, not drawn by hand.
- In M10's first half the video starts 10 frames (400 ms) before the annotation.
  Video frames 1 to 10 are present as images with `is_labeled` and all `has_labeled_*`
  flags `false` and have no annotations, so `image_id` and `file_name` still equal the video
  frame. The SoccerNet GameState evaluator ignores such images, so predictions on them are
  neither matched nor counted as false positives. Their ball rows are NaN.

**Unique ids.** `info.id` is `901` to `920` (one per half, in match order, first half first).

- `image_id` is `"3"` + `info.id` + the 6-digit video frame, for example `"3919000011"`;
  `file_name` is still `<frame:06d>.jpg`.
- An object annotation's `id` is its `image_id` + a 2-digit rank within the frame (`01`, `02`,
  ...); a pitch annotation's `id` is its `image_id` + `P`, and its `video_id` is `info.id`.
- Halves can therefore be combined in one split: TrackLab, for example, indexes images and
  pitch annotations by `image_id` across all sequences of a split.

**Placeholder objects removed.** 718 object annotations whose `bbox_pitch` values were all
`null` are removed; their images stay. The SoccerNet GameState evaluator failed on them. They were
9 whole placeholder frames (the first frame of 117092 1st and the last frames of 132831 1st,
132831 2nd, 132877 1st and 132877 2nd), which now have no objects; 500 records in the last 25
frames of some tracks in 132831 2nd and 132877 2nd, where the box track ran past the pitch track; and
21 isolated records. Removed per half:
117092 1st (25), 118578 2nd (8), 128058 2nd (10), 132831 1st (22), 132831 2nd (238), 132877 1st (44), 132877 2nd (371).

**Pitch lines renormalised.** The pitch `lines` of every file are now normalised by the
file's own `images[].width/height`. In v1.1 they were still normalised by the old nominal
3840×1504 while width and height gave the real video size.

**M9 (132831) calibration regenerated.** `raw/132831/132831_keypoints.json` carries the corrected
keypoints (two transposed labels, (88.5, 13.84) and (105, 54.16), swapped back), and
`132831_camera_intrinsics.npz`, `_mapx.npy`, `_mapy.npy` and `_homography.npy` are refitted from them
(fisheye RMS 1,260.95 px before, 18.07 px now; the previous homography was degenerate). The
homography is fitted at a 5 px RANSAC threshold rather than the 3 px used for the other matches,
because at 3 px it places the halfway line about 190 px off near the camera. With the same
undistortion settings as the other matches, M9's pitch fills a smaller part of the undistorted
frame (about 830 px wide, against about 2,900 px for the 4096-wide matches).

**Ball-action frame.** Every event has an integer `frame`: the 1-based video frame (= game-state
frame = ball row) the ball-action spotting benchmark assigns it. It is the frame nearest to `position`, that is
floor(`position` / 40 + 0.5) + 1, except for 392 events that lie exactly halfway between two frames
(`position` mod 40 = 20): for those `frame` is the later frame in 205 cases and the
earlier in 187, because the benchmark rounds on the provider's clock. Use `frame` rather
than recomputing it.

## Changes from v1.0

All of v1.1 is kept:

- Annotation frame *k* is video frame *k* in every half. Annotated frames before the first or
  after the last video frame were dropped; no per-half offset is needed.
- `images[].width/height` give the real video size (v1.0 said 3840×1504 everywhere).
  `info.name` is `CLPD-<match>-<half>`; `seq_length`, `clip_*` and `game_time_*` describe the
  whole half.
- Ball arrays have exactly one row per game-state frame (`frame` = 1..N). Frames without a
  provider value have `x = y = NaN` and `status = -1`.
- Ball-action `position` is in ms from the first frame of the half's video (v1.0: the
  provider's match clock), `gameTime` is `"<half> - m:ss"` on that clock, and the top-level
  field `"clock"` states it. Only events with both video and game state are kept: the
  unfilmed third period of 117092 (M1), 132831 (M9) and 132877 (M10) and events outside the
  annotated frames are removed.
  One event of 118578 (M6) lies 7 ms before the first frame; its `position` is 0.
- No pitch position, identity, attribute or event label was edited in either release. Box
  coordinates were edited only for M10 in v1.2.

## Time and frame convention

- All streams run at 25 fps. Video frame *k* (1-based) is game-state frame *k*, ball row
  `frame == k`, and ball-action `frame == k`; it is shown at (*k* − 1) × 40 ms.
- Times are milliseconds after the **first decoded frame** of the half's video, not after the
  mp4's time zero. In these 14 halves the mp4 video stream starts at 1.000 s:
  117092 2nd, 117093 1st, 118576 1st, 118576 2nd, 118577 2nd, 118578 1st, 128057 1st, 128057 2nd, 128058 1st, 128058 2nd, 132831 1st, 132831 2nd, 132877 1st, 132877 2nd.
  In those files, seeking by timestamp (for example `ffmpeg -ss`) lands 25 frames early unless
  1.000 s is added. Decoding by frame index is unaffected.
- Measured against detector output, annotation and video agree within 1 frame in 16 halves;
  118576 2nd and 132831 2nd are 3 frames off, 118577 2nd and 128058 2nd are 2 frames off, and
  117092 could not be measured. So alignment is good to about 0.12 s (3 frames) in every half.

## Facts worth knowing

- **Objects per frame.** 98.7% of labelled frames hold exactly 22 objects
  (players and goalkeepers; no ball, no referees).
  17,799 frames hold 14 to 21, 287 frames hold 23 to 25, and 9 hold none.
- **Boxes past the frame edge.** 1,068 boxes in 8 halves extend beyond the
  image; they are left as annotated.
- **Ball track.** The ball position is interpolated, not measured in every frame: it is
  anchored at annotated ball events and moves in straight lines at constant speed between
  anchors (in about 60% of intervals between consecutive `bas/` events the ball moves in one
  straight constant-speed segment; the rest contain further anchors). Like
  player positions, it is rounded to a 1.05 m × 0.68 m grid, so slow segments look stepped.

## Counts

| Match | 1st half frames | 2nd half frames | objects | ball-action events |
|---|---:|---:|---:|---:|
| 117092 (M1) | 67,375 | 67,549 | 2,967,010 | 2,085 |
| 117093 (M2) | 67,625 | 70,475 | 3,038,090 | 2,252 |
| 118575 (M3) | 71,850 | 73,425 | 3,187,495 | 2,117 |
| 118576 (M4) | 72,825 | 72,775 | 3,200,207 | 2,129 |
| 118577 (M5) | 72,425 | 72,475 | 3,182,525 | 1,945 |
| 118578 (M6) | 70,350 | 70,975 | 3,109,075 | 2,106 |
| 128057 (M7) | 67,625 | 70,350 | 3,035,450 | 1,932 |
| 128058 (M8) | 68,975 | 67,750 | 3,008,271 | 2,144 |
| 132831 (M9) | 67,775 | 68,375 | 2,994,702 | 2,440 |
| 132877 (M10) | 67,700 | 69,350 | 3,014,445 | 2,278 |
| **All** | 694,525 | 703,499 | 30,737,270 | 21,428 |

Frame counts include the 10 unlabelled leading frames of M10's first half.
Every video frame has a game-state image except the last video frame of 117092 2nd, where the video runs past the annotation.
4 of the 21,432 events of the v1.0 benchmark fall before the first video frame and are not in the files: 117092 (M1) half 2 PASS, 118576 (M4) half 1 PASS, 118576 (M4) half 2 PASS, 118577 (M5) half 2 PASS. Five first- or second-half events that were already outside that benchmark are also absent: the kick-off passes of 132831 (M9) and 132877 (M10) in both halves, and one M9 first-half DRIVE after the last frame.

## Known issues, left as they are

- Exact duplicate ball-action events: M7 1, M9 3, M10 1.
- M10's boxes are model-based (see above): they sit on the players, but their size is a
  typical person size at that image position, not the individual player's.
