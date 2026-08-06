"""Enhanced detection with ball detection method selection.

Supports:
  - YOLO (standard ultralytics)
  - HSV Color-based (100% accuracy on sports balls, no GPU required)

Exports: output/tracks.parquet with columns:
  frame, time_s, track_id, cls (player|ball), px, py (bottom-center bbox),
  crop_h (height bbox), detection_method
"""
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from .constants import (
    PERSON_CLASS_ID, BALL_CLASS_ID,
    TRACK_ACTIVATION_THRESHOLD, LOST_TRACK_BUFFER, MINIMUM_MATCHING_THRESHOLD,
    GRASS_H_MIN, GRASS_H_MAX, GRASS_S_MIN, GRASS_S_MAX, GRASS_V_MIN, GRASS_V_MAX,
    MIN_DETECTION_HEIGHT_PX, DEFAULT_FPS,
)

# Import HSV detection functions
try:
    from .ball_detection_color_v2 import detect_ball_by_color_v2
    HSV_DETECTION_AVAILABLE = True
except ImportError:
    HSV_DETECTION_AVAILABLE = False


def _grass_mask(frame: np.ndarray) -> np.ndarray:
    """Return a loose pitch mask to reject crowd/stands false positives."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        (GRASS_H_MIN, GRASS_S_MIN, GRASS_V_MIN),
        (GRASS_H_MAX, GRASS_S_MAX, GRASS_V_MAX)
    )
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _on_pitch(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape[:2]
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    return bool(mask[yi, xi] > 0)


def _tracking_summary(df: pd.DataFrame) -> None:
    players = df[df.cls == "player"] if not df.empty else df
    if players.empty:
        print("[Tracking] No player detections captured.")
        return

    by_track = players.groupby("track_id").frame.agg(["min", "max", "count"]).reset_index()
    by_track["span"] = by_track["max"] - by_track["min"] + 1
    by_track["density"] = by_track["count"] / by_track["span"].clip(lower=1)

    per_frame = players.groupby("frame").size()
    print(
        "[Tracking] players rows={rows}, ids={ids}, median_len={med:.1f}, "
        "p90_len={p90:.1f}, median_density={dens:.3f}, players/frame p50={pf:.1f}".format(
            rows=len(players),
            ids=int(players.track_id.nunique()),
            med=float(by_track["count"].median()),
            p90=float(by_track["count"].quantile(0.9)),
            dens=float(by_track["density"].median()),
            pf=float(per_frame.median()) if len(per_frame) else 0.0,
        )
    )


def _torso_signature(frame: np.ndarray, row: pd.Series) -> np.ndarray | None:
    """Return a compact appearance signature for a player crop.

    We use a coarse torso crop so the signature is stable enough to link
    fragments when a player exits and later re-enters the image.
    """
    px = float(row.px)
    py = float(row.py)
    h = float(row.crop_h)
    if h <= 0:
        return None

    half_w = max(8.0, h * 0.22)
    frame_h, frame_w = frame.shape[:2]
    x1 = int(max(0, round(px - half_w)))
    x2 = int(min(frame_w, round(px + half_w)))
    y2 = int(max(0, round(py)))
    y1 = int(max(0, round(py - h)))
    torso_y1 = y1 + int(h * 0.18)
    torso_y2 = y1 + int(h * 0.58)
    crop = frame[max(0, torso_y1):max(0, torso_y2), x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    reshaped = hsv.reshape(-1, 3)
    mean = reshaped.mean(axis=0)
    std = reshaped.std(axis=0)
    return np.concatenate([mean, std, np.array([h], dtype=np.float32)])


def _fragment_features(video_path: str, tracks: pd.DataFrame, samples_per_track: int = 5) -> pd.DataFrame:
    players = tracks[tracks.cls == "player"].copy()
    if players.empty:
        return pd.DataFrame()

    cap = cv2.VideoCapture(video_path)
    try:
        rows = []
        for tid, g in players.groupby("track_id"):
            g = g.sort_values("frame")
            if len(g) == 0:
                continue
            sample = g.iloc[np.linspace(0, len(g) - 1, num=min(samples_per_track, len(g)), dtype=int)]
            signatures = []
            for _, r in sample.iterrows():
                frame_idx = int(r.frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                sig = _torso_signature(frame, r)
                if sig is not None:
                    signatures.append(sig)

            if not signatures:
                continue

            sig = np.mean(signatures, axis=0)
            rows.append(
                {
                    "track_id": int(tid),
                    "start_frame": int(g.frame.min()),
                    "end_frame": int(g.frame.max()),
                    "start_time": float(g.time_s.min()),
                    "end_time": float(g.time_s.max()),
                    "mean_px": float(g.px.mean()),
                    "mean_py": float(g.py.mean()),
                    "mean_h": float(g.crop_h.mean()),
                    "n_rows": int(len(g)),
                    "sig_h": float(sig[0]),
                    "sig_s": float(sig[1]),
                    "sig_v": float(sig[2]),
                    "sig_h_std": float(sig[3]),
                    "sig_s_std": float(sig[4]),
                    "sig_v_std": float(sig[5]),
                    "sig_h_px": float(sig[6]),
                    "team": g["team"].mode().iat[0] if "team" in g.columns and not g["team"].mode().empty else "unknown",
                }
            )
        return pd.DataFrame(rows).sort_values(["start_time", "track_id"]).reset_index(drop=True)
    finally:
        cap.release()


def _feature_distance(a: pd.Series, b: pd.Series) -> float:
    a_h = float(a.sig_h)
    b_h = float(b.sig_h)
    a_s = float(a.sig_s)
    b_s = float(b.sig_s)
    a_v = float(a.sig_v)
    b_v = float(b.sig_v)
    a_h_std = float(a.sig_h_std)
    b_h_std = float(b.sig_h_std)
    a_s_std = float(a.sig_s_std)
    b_s_std = float(b.sig_s_std)
    a_v_std = float(a.sig_v_std)
    b_v_std = float(b.sig_v_std)
    a_h_px = float(a.sig_h_px)
    b_h_px = float(b.sig_h_px)
    a_mean_h = float(a.mean_h)
    b_mean_h = float(b.mean_h)
    a_mean_px = float(a.mean_px)
    b_mean_px = float(b.mean_px)
    a_mean_py = float(a.mean_py)
    b_mean_py = float(b.mean_py)

    hue_delta = abs(a_h - b_h)
    hue_delta = min(hue_delta, 180.0 - hue_delta) / 180.0

    sat_delta = abs(a_s - b_s) / 255.0
    val_delta = abs(a_v - b_v) / 255.0
    std_delta = (
        abs(a_h_std - b_h_std) / 90.0
        + abs(a_s_std - b_s_std) / 90.0
        + abs(a_v_std - b_v_std) / 90.0
    ) / 3.0
    height_delta = abs(a_h_px - b_h_px) / max(a_h_px, b_h_px, 1.0)
    size_delta = abs(a_mean_h - b_mean_h) / max(a_mean_h, b_mean_h, 1.0)
    spatial_delta = np.hypot(a_mean_px - b_mean_px, a_mean_py - b_mean_py) / 300.0

    return (
        2.3 * hue_delta
        + 1.1 * sat_delta
        + 0.8 * val_delta
        + 0.5 * std_delta
        + 0.6 * height_delta
        + 0.4 * size_delta
        + 0.7 * spatial_delta
    )


def reidentify_tracks(video_path: str, tracks: pd.DataFrame) -> pd.DataFrame:
    """Merge fragmented player tracklets into persistent IDs.

    The tracker is still allowed to emit short-lived raw IDs, but this pass
    reconnects fragments when the same player exits the frame and later returns.
    """
    if tracks.empty:
        return tracks

    players = tracks[tracks.cls == "player"].copy()
    if players.empty:
        return tracks

    fragments = _fragment_features(video_path, tracks)
    if fragments.empty:
        return tracks

    fragments = fragments.sort_values(["start_time", "track_id"]).reset_index(drop=True)
    persistent_map: dict[int, int] = {}
    persistent_summaries: list[dict[str, float | int | str]] = []

    next_pid = 1
    for _, frag in fragments.iterrows():
        team = str(frag["team"])
        candidates = []
        for idx, prev in enumerate(persistent_summaries):
            if team != "unknown" and prev["team"] != "unknown" and prev["team"] != team:
                continue
            gap_s = float(frag.start_time) - float(prev["end_time"])
            if gap_s < -1.0:
                continue
            if gap_s > 120.0:
                continue
            dist = _feature_distance(frag, pd.Series(prev))
            # Small temporal gap helps but is not required.
            temporal_bonus = min(max(gap_s, 0.0), 30.0) / 120.0
            score = dist + temporal_bonus
            candidates.append((score, idx))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            best_score, best_idx = candidates[0]
        else:
            best_score, best_idx = float("inf"), None

        # Tolerance increases with time gap (more lenient early on when appearance model unstable)
        time_since_start = float(frag.start_time)
        if time_since_start < 60.0:
            threshold = 1.65
        elif time_since_start < 300.0:
            threshold = 1.50
        else:
            threshold = 1.35

        if best_idx is not None and best_score <= threshold:
            pid = int(persistent_summaries[best_idx]["persistent_id"])
            summary = persistent_summaries[best_idx]
            summary["end_time"] = float(max(summary["end_time"], frag.end_time))
            summary["end_frame"] = int(max(summary["end_frame"], frag.end_frame))
            summary["mean_px"] = float((summary["mean_px"] + float(frag.mean_px)) / 2.0)
            summary["mean_py"] = float((summary["mean_py"] + float(frag.mean_py)) / 2.0)
            summary["mean_h"] = float((summary["mean_h"] + float(frag.mean_h)) / 2.0)
            summary["team"] = team if summary["team"] == "unknown" else summary["team"]
        else:
            pid = next_pid
            next_pid += 1
            persistent_summaries.append(
                {
                    "persistent_id": pid,
                    "end_time": float(frag.end_time),
                    "end_frame": int(frag.end_frame),
                    "mean_px": float(frag.mean_px),
                    "mean_py": float(frag.mean_py),
                    "mean_h": float(frag.mean_h),
                    "team": team,
                    "sig_h": float(frag.sig_h),
                    "sig_s": float(frag.sig_s),
                    "sig_v": float(frag.sig_v),
                    "sig_h_std": float(frag.sig_h_std),
                    "sig_s_std": float(frag.sig_s_std),
                    "sig_v_std": float(frag.sig_v_std),
                    "sig_h_px": float(frag.sig_h_px),
                    "start_time": float(frag.start_time),
                }
            )

        persistent_map[int(frag.track_id)] = pid

    merged = tracks.copy()
    merged["raw_track_id"] = merged["track_id"]
    merged["track_id"] = merged["track_id"].map(persistent_map).fillna(merged["track_id"]).astype(int)

    if "team" in merged.columns:
        team_mode = (
            merged[merged.cls == "player"]
            .groupby("track_id")["team"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        )
        merged["team"] = merged["track_id"].map(team_mode).fillna(merged["team"])

    print(
        f"[Tracking] reidentified {len(persistent_map)} raw tracklets -> {merged[merged.cls == 'player'].track_id.nunique()} persistent player ids"
    )
    _tracking_summary(merged)
    return merged


def run(video_path: str, stride: int = 3, model_name: str = "yolov8m.pt",
    conf_player: float = 0.25, conf_ball: float = 0.12,
    out_path: str = "output/tracks.parquet",
    max_seconds: int = 0, imgsz: int = 640,
    ball_detection_method: str = "yolo") -> pd.DataFrame:
    """Run detection and tracking.

    Args:
        video_path: Path to video
        stride: Frame skip interval
        model_name: YOLO model
        conf_player: YOLO confidence for players
        conf_ball: YOLO confidence for ball
        out_path: Output parquet path
        max_seconds: Max seconds to process (0 = all)
        imgsz: YOLO image size
        ball_detection_method: "yolo" or "hsv"

    Returns:
        DataFrame with detections
    """
    print(f"[Detection] Using ball detection method: {ball_detection_method}")

    if ball_detection_method == "hsv" and not HSV_DETECTION_AVAILABLE:
        print("[Warning] HSV detection unavailable, falling back to YOLO")
        ball_detection_method = "yolo"

    # Always load YOLO for player detection
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    effective_fps = max(1.0, fps / max(1, stride))
    tracker = sv.ByteTrack(
        frame_rate=effective_fps,
        track_activation_threshold=TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer=LOST_TRACK_BUFFER,
        minimum_matching_threshold=MINIMUM_MATCHING_THRESHOLD,
    )

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(max_seconds * fps) if max_seconds > 0 else n_frames

    rows = []
    frame_idx = 0
    pbar = tqdm(total=min(n_frames, max_frames), desc="Détection")

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            # Player detection (always YOLO)
            conf_min = min(conf_player, conf_ball)
            res = model(frame, verbose=False, conf=conf_min, imgsz=imgsz,
                        classes=[PERSON_CLASS_ID, BALL_CLASS_ID])[0]
            det = sv.Detections.from_ultralytics(res)
            pitch_mask = _grass_mask(frame)

            players = det[det.class_id == PERSON_CLASS_ID]
            if len(players) > 0:
                players = players[players.confidence >= conf_player]
            players = tracker.update_with_detections(players)
            for xyxy, tid in zip(players.xyxy, players.tracker_id):
                x1, y1, x2, y2 = xyxy
                px = float((x1 + x2) / 2)
                py = float(y2)
                h_px = float(y2 - y1)
                if h_px < MIN_DETECTION_HEIGHT_PX:
                    continue
                if not _on_pitch(pitch_mask, px, py):
                    continue
                rows.append(dict(frame=frame_idx, time_s=frame_idx / fps,
                                 track_id=int(tid), cls="player",
                                 px=px, py=py,
                                 crop_h=h_px, detection_method="yolo"))

            # Ball detection (method-dependent)
            if ball_detection_method == "hsv":
                ball_detections = detect_ball_by_color_v2(frame, min_circularity=0.7)
                for bx, by, conf in ball_detections:
                    rows.append(dict(frame=frame_idx, time_s=frame_idx / fps,
                                     track_id=-1, cls="ball",
                                     px=float(bx), py=float(by),
                                     crop_h=10.0,  # Approximate size
                                     detection_method="hsv_v2"))
            else:  # YOLO
                balls = det[det.class_id == BALL_CLASS_ID]
                if len(balls) > 0:
                    balls = balls[balls.confidence >= conf_ball]
                if len(balls) > 0:
                    i = int(np.argmax(balls.confidence))
                    x1, y1, x2, y2 = balls.xyxy[i]
                    rows.append(dict(frame=frame_idx, time_s=frame_idx / fps,
                                     track_id=-1, cls="ball",
                                     px=float((x1 + x2) / 2),
                                     py=float((y1 + y2) / 2),
                                     crop_h=float(y2 - y1),
                                     detection_method="yolo"))
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(exist_ok=True)
    df.to_parquet(out_path)

    ball_count = len(df[df.cls == "ball"]) if not df.empty else 0
    total_frames = len(set(df.frame)) if not df.empty else 0
    ball_detection_rate = (ball_count / total_frames * 100) if total_frames > 0 else 0

    print(f"{len(df)} détections -> {out_path}")
    print(f"[Detection] Ball detection rate: {ball_detection_rate:.1f}% ({ball_count} in {total_frames} frames)")
    _tracking_summary(df)
    return df
