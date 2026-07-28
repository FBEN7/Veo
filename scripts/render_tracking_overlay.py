"""Render an annotated MP4 from tracking parquet.

Overlays:
  - Track bboxes with track IDs
  - Colored trails per track
  - Ball marker
  - Optional speed estimate per track (pixel-domain)

Usage:
  python scripts/render_tracking_overlay.py \
      --video data/match.mp4 \
      --tracks output/tracks_teams.parquet \
      --out output/tracking_overlay.mp4
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd


def color_for_track(track_id: int) -> tuple[int, int, int]:
    # Stable pseudo-random color by track id.
    rng = np.random.default_rng(track_id * 7919)
    bgr = rng.integers(40, 240, size=3)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x2 = x + tw + 6
    y2 = y - th - 8
    cv2.rectangle(frame, (x, y), (x2, y2), color, -1)
    cv2.putText(frame, text, (x + 3, y - 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def estimate_bbox(px: float, py: float, crop_h: float) -> tuple[int, int, int, int]:
    # Approximate person bbox from bottom-center and crop height.
    h = max(int(crop_h), 12)
    w = max(int(h * 0.42), 8)
    x1 = int(px - w / 2)
    y2 = int(py)
    y1 = y2 - h
    x2 = x1 + w
    return x1, y1, x2, y2


def build_player_number_map(tracks: pd.DataFrame) -> dict[int, str]:
    """Assign stable display numbers to the 11 most present players per team.

    This produces labels like A1..A11 and B1..B11 so the overlay stays readable
    even when raw tracking IDs are fragmented across a match.
    """
    if "team" not in tracks.columns:
        return {}

    players = tracks[(tracks.cls == "player") & (tracks.team.isin(["team_A", "team_B"]))].copy()
    if players.empty:
        return {}

    stats = (
        players.groupby(["team", "track_id"])
        .agg(first_frame=("frame", "min"), frames=("frame", "size"))
        .reset_index()
        .sort_values(["team", "frames", "first_frame"], ascending=[True, False, True])
    )

    mapping: dict[int, str] = {}
    prefixes = {"team_A": "A", "team_B": "B"}
    for team, group in stats.groupby("team", sort=False):
        prefix = prefixes.get(team, team.replace("team_", ""))
        for number, (_, row) in enumerate(group.head(11).iterrows(), start=1):
            mapping[int(row.track_id)] = f"{prefix}{number}"

    return mapping


def draw_player_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x1 = max(x, 0)
    y1 = max(y - th - 8, 0)
    x2 = min(x1 + tw + 8, frame.shape[1] - 1)
    y2 = min(y + 2, frame.shape[0] - 1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    cv2.putText(frame, text, (x1 + 4, y2 - 5), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/match.mp4")
    ap.add_argument("--tracks", default="output/tracks_teams.parquet")
    ap.add_argument("--out", default="output/tracking_overlay.mp4")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="0 = full video")
    ap.add_argument("--trail-len", type=int, default=30)
    ap.add_argument("--show-speed", action="store_true")
    args = ap.parse_args()

    tracks = pd.read_parquet(args.tracks).sort_values(["frame", "track_id"]).reset_index(drop=True)
    player_numbers = build_player_number_map(tracks)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_seconds > 0:
        max_frames = min(max_frames, int(args.max_seconds * fps))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width, height))

    by_frame = {int(f): g for f, g in tracks.groupby("frame")}
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.trail_len))
    last_pos: dict[int, tuple[float, float, float]] = {}

    frame_idx = 0
    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        g = by_frame.get(frame_idx)
        if g is not None:
            # First pass: update trails.
            for _, r in g.iterrows():
                tid = int(r.track_id)
                px = float(r.px)
                py = float(r.py)
                trails[tid].append((int(px), int(py)))

            # Draw trails.
            for _, r in g.iterrows():
                tid = int(r.track_id)
                cls = str(r.cls)
                if cls != "player":
                    continue
                pts = list(trails[tid])
                if len(pts) >= 2:
                    col = color_for_track(tid)
                    cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, col, 2)

            # Draw objects.
            for _, r in g.iterrows():
                tid = int(r.track_id)
                cls = str(r.cls)
                px = float(r.px)
                py = float(r.py)

                if cls == "ball":
                    cv2.circle(frame, (int(px), int(py)), 7, (0, 255, 255), 2)
                    draw_label(frame, "ball", int(px) + 8, int(py) + 18, (0, 160, 160))
                    continue

                crop_h = float(r.crop_h) if pd.notna(r.crop_h) else 30.0
                x1, y1, x2, y2 = estimate_bbox(px, py, crop_h)
                col = color_for_track(tid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

                number_label = player_numbers.get(tid)
                if number_label is not None:
                    label = number_label
                else:
                    label = f"ID {tid}"
                if args.show_speed:
                    prev = last_pos.get(tid)
                    speed_txt = ""
                    if prev is not None:
                        ppx, ppy, pt = prev
                        dt = max((frame_idx / fps) - pt, 1e-3)
                        v = np.sqrt((px - ppx) ** 2 + (py - ppy) ** 2) / dt
                        speed_txt = f" {v:4.1f}px/s"
                    label += speed_txt
                    last_pos[tid] = (px, py, frame_idx / fps)

                draw_player_label(frame, label, max(x1, 2), max(y1 - 2, 20), col)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"[Overlay] Saved: {args.out}")


if __name__ == "__main__":
    main()
