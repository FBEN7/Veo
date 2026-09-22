"""What a homography would need here, and why it is not yet possible.

Every metric figure this pipeline reports -- distance covered, top speed,
sprint counts, pass length -- divides pixels by a single `px_per_m` taken
from the median player height. A homography would replace that with a proper
ground plane. This probes whether one can be built from the footage in hand.

Four measurements, in the order they were taken.

    python probe_homography.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.detect_track_hybrid import _grass_mask

CLIPS = (("Veo", "output_veo"),
         ("SoccerNet w1", "output_soccernet"),
         ("SoccerNet w2", "output_soccernet_w2"))

# A footballer, for turning pixel heights into metres.
PLAYER_HEIGHT_M = 1.75


def scale_by_image_row(bands: int = 6):
    """How much the metre-per-pixel scale varies down the frame.

    This is the finding that matters. A single `px_per_m` is only correct at
    one depth; under perspective the scale changes with image row, and here
    it changes by a factor of three.
    """
    print("1. Does one px_per_m describe the whole frame?\n")
    for name, d in CLIPS:
        raw = pd.read_parquet(Path(d) / "tracks_grass.parquet")
        players = raw[raw.cls == "player"]
        info = json.loads((Path(d) / "clip.json").read_text())
        prof = json.loads((Path(d) / "profile.json").read_text())
        h = info["height"]

        band = (players.py / h * bands).clip(0, bands - 1).astype(int)
        med = players.groupby(band).crop_h.median()
        implied = med / PLAYER_HEIGHT_M
        print(f"  {name}: shipped {prof['px_per_m']:.1f} px/m; "
              f"by image row {implied.min():.1f} to {implied.max():.1f} "
              f"-- a factor of {implied.max() / implied.min():.2f}")


def ground_plane_fit():
    """Player pixel height against image row.

    For a camera looking at a plane, a fixed-height object projects to a
    pixel height linear in the image row, reaching zero at the horizon. The
    slope is the player's height over the camera's, so the fit yields a
    camera height that can be sanity-checked against the real world.
    """
    print("\n2. Is the ground-plane model consistent with the detections?\n")
    for name, d in CLIPS:
        raw = pd.read_parquet(Path(d) / "tracks_grass.parquet")
        players = raw[raw.cls == "player"]
        y = players.py.to_numpy(float)
        ph = players.crop_h.to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(ph) & (ph > 5)
        y, ph = y[ok], ph[ok]

        slope, intercept = np.polyfit(y, ph, 1)
        for _ in range(5):
            resid = ph - (slope * y + intercept)
            keep = np.abs(resid) < 2.5 * resid.std()
            slope, intercept = np.polyfit(y[keep], ph[keep], 1)

        pred = slope * y + intercept
        r2 = 1 - np.sum((ph - pred) ** 2) / np.sum((ph - ph.mean()) ** 2)
        print(f"  {name}: horizon at y={-intercept / slope:7.1f}, "
              f"R^2 {r2:.2f}, implied camera height "
              f"{PLAYER_HEIGHT_M / slope:5.1f} m")
    print("\n  The camera heights are believable -- a Veo mast and a "
          "broadcast gantry --\n  so the model is real, but R^2 is low: youth "
          "players differ in height,\n  and a bounding box changes with pose "
          "and occlusion.")


def speed_anisotropy():
    """Whether depth motion is compressed relative to lateral motion.

    If a single scale were right, a player's speed would not depend on which
    way they were running. It does -- but not consistently, which is what
    rules this out as a way to pin the focal length.
    """
    print("\n3. Could the missing focal length be recovered from the "
          "motion itself?\n")
    for name, d in CLIPS[:2]:
        raw = pd.read_parquet(Path(d) / "tracks_grass.parquet")
        prof = json.loads((Path(d) / "profile.json").read_text())
        k = prof["px_per_m"]
        lat, dep = [], []
        for _, g in raw[raw.cls == "player"].groupby("track_id"):
            if len(g) < 25:
                continue
            g = g.sort_values("frame")
            dx = g.px.diff().to_numpy() / k
            dy = g.py.diff().to_numpy() / k
            dt = g.frame.diff().to_numpy() / 25.0
            ok = np.isfinite(dx) & np.isfinite(dy) & (dt > 0)
            dx, dy, dt = dx[ok], dy[ok], dt[ok]
            speed = np.hypot(dx, dy) / dt * 3.6
            ang = np.abs(np.degrees(np.arctan2(dy, dx)))
            lat.extend(speed[(ang < 25) | (ang > 155)])
            dep.extend(speed[(ang > 65) & (ang < 115)])
        a, b = np.percentile(lat, 90), np.percentile(dep, 90)
        print(f"  {name}: lateral p90 {a:5.1f} km/h, depth p90 {b:5.1f}, "
              f"ratio {a / b:.2f}")
    print("\n  Perspective compresses depth, so the ratio should exceed 1 "
          "everywhere.\n  It is 1.49 on broadcast and 0.78 on Veo. Footballers "
          "also run along the\n  pitch more than across it, and that confound "
          "is large enough to sink\n  the idea. No focal length from this.")


def marking_visibility(clip: str, frames=(1400, 2600, 3800, 4300)):
    """How much of the pitch geometry is actually visible to anchor to."""
    print("\n4. Are there pitch markings to anchor a homography to?\n")
    cap = cv2.VideoCapture(clip)
    idx, grabbed = 0, {}
    while idx <= max(frames):
        ok, f = cap.read()
        if not ok:
            break
        if idx in frames:
            grabbed[idx] = f
        idx += 1
    cap.release()

    for k, f in sorted(grabbed.items()):
        grass = _grass_mask(f)
        n, lab, st, _ = cv2.connectedComponentsWithStats(
            (grass > 0).astype(np.uint8), 8)
        if n > 1:
            big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            grass = ((lab == big) * 255).astype(np.uint8)
        grass = cv2.erode(grass, np.ones((15, 15), np.uint8), 1)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 25, -14)
        marks = cv2.bitwise_and(thr, grass)
        print(f"  frame {k:5d}: pitch fills {100 * (grass > 0).mean():3.0f}% "
              f"of the view, markings {100 * (marks > 0).mean():4.1f}% of it")
    print("\n  Markings are present -- a line or an arc in most frames -- but "
          "one or two\n  segments per frame is not four correspondences, and "
          "which line is which\n  is exactly the ambiguity a homography has to "
          "resolve.")


def main():
    scale_by_image_row()
    ground_plane_fit()
    speed_anisotropy()
    clip = json.loads(Path("output_veo/clip.json").read_text())["path"]
    if Path(clip).exists():
        marking_visibility(clip)


if __name__ == "__main__":
    main()
