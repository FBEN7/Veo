"""Metres from a camera that pans: the path `main.py --camera moving` takes.

A Veo export is the follow-cam view: a virtual camera that pans with play.
One homography for the whole clip, which is what `main.py` otherwise uses,
turns every pan into players sprinting (VEO_FOOTAGE.md, 2026-11-23: top
speeds of 33-39 km/h on a 25-s clip). This path does what
`run_veo_analysis.py` does, in one function shared by `main.py` and by the
measurement in `research/session_scratchpad/moving_cam/`:

1. camera motion (optical flow on the background), checked by its own
   validator; a refused estimate is not subtracted, and the run says so;
2. positions compensated, so they are fixed to the ground;
3. one scale in pixels per metre: the ground plane's, else player height;
4. metres = pixels / scale.

What comes out is metric but *relative*: distances and speeds mean
something, positions on the pitch do not. Anything that needs the goal or
the touchlines (heatmaps, field tilt, shots, goals, out of play) has no
footing on this path.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import camera_motion, ground_plane, pixel_scale

PLAYERS_ON_PITCH = 22


def coverage(tracks: pd.DataFrame, n_frames: int) -> float:
    """Share of the 22 players' frames in which a team-assigned player is
    detected: what any per-player figure from this footage is computed on.
    Counted in detections, not track spans: a re-identified track spans the
    gaps where its player was out of view."""
    if n_frames <= 0 or tracks is None or tracks.empty:
        return 0.0
    p = tracks[(tracks.cls == "player") & tracks.team.isin(["team_A", "team_B"])]
    return float(min(1.0, p.groupby("frame").size().clip(upper=PLAYERS_ON_PITCH).sum()
                     / (PLAYERS_ON_PITCH * n_frames)))


def to_metric(video_path: str, tracks: pd.DataFrame, work_dir: str | Path,
              verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Tracks in ground-fixed metres, and how they were obtained."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    motion_path = work_dir / "camera_motion.npy"
    if motion_path.exists():
        motion = np.load(motion_path)
    else:
        motion = camera_motion.estimate_camera_motion(str(video_path), verbose=verbose)
        np.save(motion_path, motion)
    usable, why = camera_motion.validate_motion(str(video_path), motion, verbose=verbose)
    comp = camera_motion.compensate(tracks, motion) if usable else tracks.copy()

    players = comp[comp.cls == "player"] if "cls" in comp.columns else comp
    plane = ground_plane.load_or_build(str(video_path), players, width, height,
                                       work_dir / "ground_plane.json", verbose=verbose)
    scale = ground_plane.effective_px_per_m(players, plane, verbose=verbose) if plane else None
    source = "ground plane"
    if scale is None:
        scale = pixel_scale.estimate_px_per_m(comp)
        source = "player height"
    metric = pixel_scale.to_metric_coords(comp, scale)
    info = {"compensated": bool(usable), "motion check": why,
            "px_per_m": round(float(scale), 2), "scale from": source}
    if verbose:
        print(f"  [moving] camera motion {'compensated' if usable else 'NOT compensated'} "
              f"({why}); scale {scale:.1f} px/m from {source}")
    return metric, info
