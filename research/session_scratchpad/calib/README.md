---
license: cc-by-4.0
pretty_name: SoccerTrack v2
task_categories:
  - object-detection
  - video-classification
tags:
  - soccer
  - sports-analytics
  - multi-object-tracking
  - game-state-reconstruction
  - action-spotting
---

# SoccerTrack v2

Ten university-level soccer matches (934 minutes) recorded by fixed panoramic
camera systems whose field of view spans the entire pitch, released together
with frame-level game state annotations and player-linked ball action events
on the same footage.

## Contents

| Folder | Contents |
|---|---|
| `videos/` | 20 panoramic half-match videos (two 45-minute periods per match) |
| `gsr/` | Game state annotations: one JSON per half (pitch coordinates, jersey numbers and persistent player identifiers, roles, team sides) |
| `bas/` | Ball action events: twelve classes per match, with acting team and actor identifiers shared with the game state annotations; `position` is milliseconds from the first decoded frame of the half's video, and `frame` is the video (and game state) frame the event falls on |
| `ball/` | Ball track per half, one row per game state frame (auxiliary annotation). It is a reconstruction rather than a per-frame measurement: it is anchored at the annotated events and interpolated in straight constant-speed lines between anchors, so it carries event timing. Positions are clamped to the pitch rectangle in matches 132831 and 132877 |
| `mot/` | Curated bounding-box subset used for the SoccerTrack Challenge at MMSports 2025: per-match ground-truth files and the paired 4-minute clips |
| `raw/` | Per-match calibration files (65 hand-annotated pitch keypoints per match, distortion maps, camera intrinsics) and the provider's raw tracking and event exports, with personal names removed |

## Version

This is **release v1.2** (tag `v1.2`); earlier releases remain available as tags `v1.1`
and `v1.0`. In v1.2 every annotation lines up with the released video frame for frame:
game state frame *k* is video frame *k* in every half, ball action events carry their
frame, and times are milliseconds after the first decoded frame of each half's video.
The unfilmed third period that three matches played, and the few events that fall
outside the filmed frames, are removed, leaving 21,428 ball action events. Ids are
unique across all twenty halves, so halves can be loaded together. `CHANGES_v1_2.md`
lists every change; the main fix over v1.1 is the synchronisation of match 132877 (M10),
whose positions were about one second out of step with its video.

## Match labels

Files are named by the provider's match identifiers. The paper labels the
matches M1 to M10 in ascending identifier order:

| Paper label | Match id | Split |
|---|---|---|
| M1 | 117092 | train |
| M2 | 117093 | validation |
| M3 | 118575 | train |
| M4 | 118576 | train |
| M5 | 118577 | train |
| M6 | 118578 | train |
| M7 | 128057 | test |
| M8 | 128058 | train |
| M9 | 132831 | test |
| M10 | 132877 | validation |

Benchmark split (SoccerTrack Challenge 2025): test M7 + M9, validation
M2 + M10, train the remaining six. The paper's ball action spotting results
use five-fold cross-match validation in which fold 0 tests on M7 + M9.

## Coordinate convention

Pitch coordinates are metric on a 105 m x 68 m pitch with the origin at the
centre circle; x increases towards the right half of the pitch as seen from
the main camera and y towards the touchline nearer the camera.

## Code and documentation

Loaders, evaluation CLIs (GS-HOTA, BAS mAP, MOT HOTA) and baseline starter
kits: https://github.com/AtomScott/SoccerTrack-v2

## License

CC BY 4.0. The release is pseudonymised: players are identified by jersey
numbers and integer identifiers, and no personal names are included. Please cite the SoccerTrack v2 paper (in submission; citation
to be added) when publishing results.
