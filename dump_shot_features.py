"""Candidate-strike readings for the shot classifier, without the full pipeline.

`score_hand_labels.py --dump-features` needs a clip already run through
`score_soccernet.py`: player tracking with yolov8m on every frame, team
colours, re-identification. On the new SoccerNet windows that ran at 2.7 to
4 s a frame -- over an hour a clip -- and the classifier uses none of it.
It needs the ball, which the fine-tuned detector finds on its own
(`src/ball_detector.py`); where the players stand at the strike, for how
crowded it is, which yolov8n at 640 gives on a sample of frames; and the
goal's pose, which `score_hand_labels.goal_maps` builds from the corners.

So this prepares a minimal output directory -- the clip's identity, the
players every `PLAYER_STRIDE` frames, the ball candidates -- and goes
straight to the readings. A clip that already has a full pipeline run keeps
its own tracks, and the COCO ball rows fill frames the fine-tuned detector
leaves empty, as in the measured best configuration.

    python dump_shot_features.py <labels.txt> --out output_<clip> \\
        --video <clip.mp4> --features feat/<clip>.parquet \\
        --corners goal_corners.json --goal-weights <box detector> \\
        --goal-keypoints .cache/goal_keypoints \\
        --ball-detectors .cache/ball_detector --camera-store camera.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

# Players on every n-th frame: crowding is read at the strike, and the
# nearest sampled frame is a tenth of a second away at most.
PLAYER_STRIDE = 3
PLAYER_CONF = 0.08


def players(info: dict, stride: int = PLAYER_STRIDE) -> pd.DataFrame:
    """People on every `stride`-th frame, at the foot point, as the
    pipeline's tracks give them."""
    import cv2
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(info["path"])
    rows, index = [], -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        index += 1
        if index % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        res = model(frame, verbose=False, conf=PLAYER_CONF, imgsz=640,
                    classes=[0])[0]
        for (x0, y0, x1, y1), score in zip(res.boxes.xyxy.cpu().numpy(),
                                           res.boxes.conf.cpu().numpy()):
            for f in range(index, index + stride):
                # Spread over the frames it stands for, so a lookup by the
                # strike's frame finds it.
                rows.append({"frame": f, "time_s": f / info["fps"],
                             "track_id": -1, "cls": "player",
                             "px": float(x0 + x1) / 2, "py": float(y1),
                             "crop_h": float(y1 - y0),
                             "detection_method": "yolov8n",
                             "confidence": float(score)})
    cap.release()
    return pd.DataFrame(rows)


def prepare(out_dir: Path, video: str, weights: Path) -> dict:
    """clip.json, tracks.parquet and the ball candidates, where missing."""
    from score_soccernet import probe_clip
    from src.ball_detector import run_clip

    out_dir.mkdir(parents=True, exist_ok=True)
    info_path = out_dir / "clip.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
    else:
        info = probe_clip(video)
        info_path.write_text(json.dumps(info, indent=1))
    tracks = out_dir / "tracks.parquet"
    if not tracks.exists():
        t = time.time()
        table = players(info)
        table.to_parquet(tracks)
        print(f"  [players] {len(table)} rows in {time.time() - t:.0f} s")
    t = time.time()
    run_clip(out_dir, info, weights)
    print(f"  [ball] candidates ready in {time.time() - t:.0f} s")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", help="needed when --out is not prepared yet")
    ap.add_argument("--features", required=True)
    ap.add_argument("--corners", required=True)
    ap.add_argument("--goal-weights", required=True)
    ap.add_argument("--goal-keypoints")
    ap.add_argument("--auto-corners",
                    help="corner model weights: find the corners of a clip "
                         "nobody clicked and add them to --corners "
                         "(auto_goal_corners.py)")
    ap.add_argument("--goal-conf", type=float)
    ap.add_argument("--ball-detectors", required=True)
    ap.add_argument("--camera-store")
    ap.add_argument("--midfield", action="store_true")
    args = ap.parse_args()
    args.old_camera = False
    args.anchor_too = False

    import detect_shots
    from score_hand_labels import goal_maps, match_name, read_labels
    from src.shot_features import candidates

    start = time.time()
    out_dir = Path(args.out)
    clip = out_dir.name.replace("output_", "")
    weights = (Path(args.ball_detectors) / f"without_{clip.split('_')[0]}"
               / "run" / "weights" / "best.pt")
    info = prepare(out_dir, args.video, weights)
    if args.auto_corners:
        from auto_goal_corners import add_clip

        t = time.time()
        add_clip(out_dir, Path(args.auto_corners), Path(args.corners))
        print(f"  [corners] found in {time.time() - t:.0f} s")
    ball = detect_shots.ball_track(out_dir, ball_detector="fill")
    t = time.time()
    maps = goal_maps(out_dir, info, ball, args)
    if maps is None:
        raise SystemExit("could not calibrate this clip from the corners")
    print(f"  [goal] poses on {len(maps)} frames in {time.time() - t:.0f} s")
    t = time.time()
    tracks = pd.read_parquet(out_dir / "tracks.parquet")
    table = candidates(ball, maps, info["fps"],
                       players=tracks[tracks.cls == "player"])
    truth, _, _ = read_labels(Path(args.labels))
    table.insert(0, "clip", clip)
    table["match"] = match_name(Path(args.labels)) or ""
    table.attrs["truth"] = json.dumps(
        [{"time_s": e["time_s"], "event_type": e["event_type"]}
         for e in truth])
    table.attrs["duration_s"] = float(ball.time_s.max())
    # What the hand-set rules find on the same ball and poses, so the
    # classifier is compared with them clip for clip.
    from src import goal_plane

    rules = goal_plane.find_shots(ball, maps, info["fps"])
    table.attrs["rule_shots"] = json.dumps(
        [{"time_s": s["time_s"], "outcome": s["outcome"]} for s in rules])
    Path(args.features).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.features)
    print(f"  [features] {len(table)} candidates in {time.time() - t:.0f} s; "
          f"{time.time() - start:.0f} s in all, to {args.features}")


if __name__ == "__main__":
    main()
