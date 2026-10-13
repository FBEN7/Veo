"""Calibration semi-automatique de l'homographie.

Workflow:
    1) Propose automatiquement 4 coins (TL, TR, BR, BL)
    2) Ajuste les points à la souris (drag)
    3) Sauvegarde l'homographie finale

Raccourcis:
    - souris gauche: déplacer le point le plus proche
    - `1..4`: sélectionner explicitement un point
    - `r`: réinitialiser avec les points auto
    - `c`: vider les points
    - `Enter`/`Space`: sauvegarder (si 4 points)
    - `q`: quitter sans sauvegarder

Usage:
    python -m src.calibrate data/match.mp4
    python -m src.calibrate data/match.mp4 --out output/homography_refined.npy
"""
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

from src import auto_calibrate

PITCH_POINTS = np.array(
    [[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32
)

points: list[tuple[int, int]] = []
selected_idx: int | None = None
dragging = False


def _clip_point(x: int, y: int, w: int, h: int) -> tuple[int, int]:
    return max(0, min(x, w - 1)), max(0, min(y, h - 1))


def _nearest_idx(x: int, y: int, pts: list[tuple[int, int]]) -> int | None:
    if not pts:
        return None
    d = [np.hypot(px - x, py - y) for px, py in pts]
    idx = int(np.argmin(d))
    return idx if d[idx] <= 80 else None


def _propose_points(video_path: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    frame = auto_calibrate._best_line_frame(video_path)
    h, w = frame.shape[:2]

    mask = auto_calibrate._white_on_green_mask(frame)
    corners = auto_calibrate._detect_corners(mask, h, w)

    if corners is not None:
        corners = auto_calibrate._order_corners(corners)
        if auto_calibrate._corners_are_plausible(corners, h, w):
            pts = [(int(x), int(y)) for x, y in corners]
            return frame, pts

    corners = auto_calibrate._fallback_corners_from_green(frame)
    corners = auto_calibrate._order_corners(corners)
    pts = [(int(x), int(y)) for x, y in corners]
    return frame, pts


def _draw_overlay(frame: np.ndarray, pts: list[tuple[int, int]], active: int | None) -> np.ndarray:
    disp = frame.copy()
    labels = ["TL", "TR", "BR", "BL"]

    if len(pts) == 4:
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(disp, [poly], True, (0, 255, 0), 2)

    for i, (x, y) in enumerate(pts):
        color = (0, 255, 255) if active == i else (0, 0, 255)
        cv2.circle(disp, (x, y), 8, color, -1)
        cv2.putText(
            disp,
            f"{i+1}:{labels[i]}",
            (x + 10, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    help_lines = [
        "Drag point with left mouse | 1..4 select point",
        "r: reset auto | c: clear | Enter/Space: save | q: quit",
    ]
    y0 = 24
    for txt in help_lines:
        cv2.putText(disp, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y0 += 24

    return disp


def on_mouse(event, x, y, flags, param):
    global selected_idx, dragging, points

    h, w = param
    x, y = _clip_point(x, y, w, h)

    if event == cv2.EVENT_LBUTTONDOWN:
        idx = _nearest_idx(x, y, points)
        if idx is None and len(points) < 4:
            points.append((x, y))
            selected_idx = len(points) - 1
            print(f"Point {len(points)}: ({x}, {y})")
        elif idx is not None:
            selected_idx = idx
            dragging = True

    elif event == cv2.EVENT_MOUSEMOVE and dragging and selected_idx is not None:
        points[selected_idx] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False


def main(video_path: str, out_path: str) -> None:
    global points, selected_idx, dragging

    try:
        frame, auto_pts = _propose_points(video_path)
    except Exception:
        cap = cv2.VideoCapture(video_path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"Impossible de lire {video_path}")
        h, w = frame.shape[:2]
        auto_pts = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]

    points = auto_pts.copy()
    selected_idx = None
    dragging = False

    h, w = frame.shape[:2]
    win = "Calibration semi-auto - ajuste les 4 points"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse, param=(h, w))
    print("[Calibrate] Points auto proposes:")
    for i, p in enumerate(points, start=1):
        print(f"  {i}: {p}")

    while True:
        disp = _draw_overlay(frame, points, selected_idx)
        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            sys.exit("Annulé.")

        if key in (13, 32):  # Enter / Space
            if len(points) == 4:
                break
            print("[Calibrate] Il faut 4 points pour sauvegarder.")

        if key == ord("r"):
            points = auto_pts.copy()
            print("[Calibrate] Reset sur points auto.")

        if key == ord("c"):
            points = []
            selected_idx = None
            print("[Calibrate] Points effaces.")

        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            idx = key - ord("1")
            if idx < len(points):
                selected_idx = idx

    cv2.destroyAllWindows()

    src = np.array(points, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, PITCH_POINTS)
    out = Path(out_path)
    out.parent.mkdir(exist_ok=True)
    np.save(out, H)
    print(f"Homographie sauvegardée -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Semi-auto pitch calibration")
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("--out", default="output/homography_refined.npy", help="Output homography path")
    args = ap.parse_args()
    main(args.video, args.out)
