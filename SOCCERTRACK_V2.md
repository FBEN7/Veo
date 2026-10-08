# SoccerTrack v2: fixed panoramic footage with ground truth

The footage this project is meant for comes from a fixed camera that sees
the whole pitch -- no cuts, no replays -- which broadcast clips (SoccerNet,
the Stoke and Reading windows) are not. Veo's panorama is that view, but
there is no public Veo panorama, and Veo's support pages say only the
follow-cam view can be downloaded. SoccerTrack v2 is the same kind of
footage, filmed by BePro's fixed panoramic systems:

- ten university matches, 934 minutes, both halves, 4K panoramas (4096
  wide for eight matches, 3840 x 1504 for one);
- **game state** per frame: every person's pitch position (metres, origin
  at the centre spot), shirt number, persistent id, role and side;
- **21,428 ball actions** in twelve classes (pass, high pass, cross,
  drive, shot, goal, header, tackle, block, out, throw-in, free kick),
  each with the side and the player who made it;
- per-match calibration; a split (train six, validation M2 and M10, test
  M7 and M9) from the SoccerTrack Challenge 2025.

CC BY 4.0, pseudonymised (numbers and integer ids, no names). Scott et
al., arXiv 2508.01802; https://huggingface.co/datasets/atomscott/soccertrack-v2.

With it the identities, the number reader and the event detector can be
measured on the kind of footage they are for, without clicking: the
ground truth already says who everyone is and who did what.

## Access

The dataset is gated: log in to Hugging Face, accept its terms on the
dataset page (approved automatically), create a read token and set it as
`HF_TOKEN` in the environment.

## Reading it

`src/soccertrack_v2.py` reads it without downloading whole files -- a
half's game-state file is ~2.7 GB of JSON and its video a 45-minute 4K
panorama:

- the game state is streamed and flattened once per half into a parquet
  table in `.cache/soccertrack_v2/gsr` (`gsr_table`);
- the events are read whole (`events`), and `with_actors` adds the actor's
  shirt and position from the game state;
- the video is read by HTTP range requests, so the decoder seeks straight
  to the minutes wanted (`RemoteFile`, `read_frames`, `cut_clip`): on a
  local clip, frames read after a seek were identical to OpenCV's, and
  2.4 MB of 24 MB was fetched for five frames.

The released files differ from the repository's own format notes (a
SoccerNet game-state file rather than flat records, the frame number
inside a string id, attributes nested, capitals in event labels, event
times counted from the match start before release v1.2); the module
docstring lists what is read. The direction of the pitch `y` axis is given
both ways in the dataset's documents; `y_towards_camera` settles it from
the data when a half is first read.

`fetch_soccertrack_v2.py` cuts a window and its ground truth into
`data/soccertrack_v2/`: the video (H.264, full resolution, so it plays in
the labelling pages), every person on every frame (`_gt.parquet`), the
ball actions with their actors (`_events.csv`) and where it came from
(`.json`).

    python fetch_soccertrack_v2.py --list
    python fetch_soccertrack_v2.py --match 117093 --half 1 --start 10 --minutes 3

`python -m src.soccertrack_v2` checks the reading on synthetic files in the
shipped format; nothing here has yet been run on the dataset itself, which
needs the token.
