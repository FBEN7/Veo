"""Derive detection parameters from the video itself, before any run.

Parameters tuned on one video do not transfer. Thresholds chosen on 1920x1080
broadcast footage left 8 players per frame of 22 on 640x360 footage, because
smaller players score lower and a fixed conf_player = 0.25 discarded them. The
same conf_ball gave ~1 candidate per frame on one video and 3.13 on another.

Football supplies the calibration targets. A match has 22 players and one
ball, a player is about 1.75 m tall. Thresholds can therefore be chosen so the
detector's output satisfies those constraints on whatever footage arrives.

What is derived:

  imgsz          smallest input size reaching most of the best ball detection
                 rate, since upscaling is what makes a small ball detectable
                 and it is the dominant cost
  conf_player    where the person-count curve flattens, within the roster
  conf_ball      lowest confidence keeping candidates near one per frame
  px_per_m       from median player bbox height against 1.75 m
  compensate_camera  whether the camera actually moves
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .constants import PERSON_CLASS_ID, BALL_CLASS_ID
from .detect_track_hybrid import _grass_mask, _near_pitch

# A match has 22 players, but the detector returns *people*: referees and
# assistants stand on the pitch too. Capping at 22 made no threshold
# satisfiable, so the rule fell through to its most aggressive value and
# discarded real players. The roster proper is enforced later by
# player_filter, once team assignment can tell players from officials.
MAX_PLAYERS_ON_PITCH = 22
MAX_PERSONS_PER_FRAME = MAX_PLAYERS_ON_PITCH + 4

# Ball candidates to aim for per frame. Above one gives the selection step
# alternatives; too many and it is choosing among noise.
TARGET_BALL_PER_FRAME = 1.5

IMGSZ_OPTIONS = (640, 960, 1280)

# Accept the smallest input size reaching this fraction of the best observed
# ball detection rate.
IMGSZ_ACCEPT_FRACTION = 0.90

CONF_GRID = (0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)

# Never choose a threshold below this. Detections under ~0.05 are
# noise-dominated whatever the counts say.
MIN_USABLE_CONF = 0.05

MIN_HEIGHT_FRACTION = 0.35
PLAYER_HEIGHT_M = 1.75

# Median frame-to-frame background movement, in pixels, above which camera
# compensation is worth attempting. Whether it is actually applied is decided
# by camera_motion.validate_motion, not here.
CAMERA_MOTION_THRESHOLD_PX = 1.0


@dataclass
class VideoProfile:
    """Detection parameters derived from a specific video."""

    video: str
    width: int
    height: int
    fps: float
    n_frames: int
    imgsz: int
    conf_player: float
    conf_ball: float
    px_per_m: float
    player_height_px: float
    ball_size_px: float
    min_detection_height_px: float
    compensate_camera: bool
    camera_motion_px: float
    players_per_frame: float
    ball_per_frame: float
    notes: list[str]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path: str | Path) -> "VideoProfile":
        return VideoProfile(**json.loads(Path(path).read_text()))

    def summary(self) -> str:
        lines = [
            f"  resolution     {self.width}x{self.height} @ {self.fps:.0f} fps",
            f"  imgsz          {self.imgsz}",
            f"  conf_player    {self.conf_player:.2f}  "
            f"-> {self.players_per_frame:.1f} players/frame",
            f"  conf_ball      {self.conf_ball:.2f}  "
            f"-> {self.ball_per_frame:.2f} ball/frame",
            f"  scale          {self.px_per_m:.1f} px/m "
            f"(player {self.player_height_px:.0f} px, "
            f"ball ~{self.ball_size_px:.1f} px)",
            f"  min height     {self.min_detection_height_px:.0f} px",
            f"  camera         "
            f"{'moves' if self.compensate_camera else 'static'} "
            f"({self.camera_motion_px:.2f} px/frame)",
        ]
        return "\n".join(lines) + ("\n" + "\n".join(f"  ! {n}" for n in self.notes)
                                   if self.notes else "")


def _sample_indices(n_frames: int, n_samples: int) -> list[int]:
    """Frames spread across the video, not just the opening seconds.

    Play at kick-off is not representative: players stand in formation and the
    ball is central.
    """
    if n_frames <= n_samples:
        return list(range(n_frames))
    return list(np.linspace(0, n_frames - 1, n_samples).astype(int))


def _probe_detections(video: str, model: YOLO, frames: list[int], imgsz: int):
    """Confidences per frame for players and ball, plus player box heights."""
    cap = cv2.VideoCapture(video)
    player_conf, ball_conf, heights = [], [], []
    for idx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        res = model(frame, verbose=False, conf=min(CONF_GRID), imgsz=imgsz,
                    classes=[PERSON_CLASS_ID, BALL_CLASS_ID])[0]
        cls = res.boxes.cls.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        xyxy = res.boxes.xyxy.cpu().numpy()

        # Count only what is on the playing surface. A crowd is made of
        # people, so unmasked counts measure attendance: the broadcast clip
        # gives 69 persons per frame at conf 0.02 and still 24 at 0.50.
        # Choosing a threshold from that suppresses spectators by raising it,
        # discarding distant players along with them.
        mask = _grass_mask(frame)
        frame_players, frame_balls = [], []
        for box, c, cf in zip(xyxy, cls, conf):
            cx = float((box[0] + box[2]) / 2)
            if c == PERSON_CLASS_ID:
                if not _near_pitch(mask, cx, float(box[3]), 12):
                    continue
                frame_players.append(cf)
                if cf >= 0.25:
                    heights.append(float(box[3] - box[1]))
            else:
                cy = float((box[1] + box[3]) / 2)
                if _near_pitch(mask, cx, cy, max(12, int(box[3] - box[1]))):
                    frame_balls.append(cf)

        player_conf.append(np.array(frame_players))
        ball_conf.append(np.array(frame_balls))
    cap.release()
    return player_conf, ball_conf, heights


def _pick_player_conf(player_conf: list[np.ndarray]) -> tuple[float, float]:
    """Confidence where the count curve flattens, within the roster bound.

    The rule used to be "lowest confidence whose median count stays within the
    roster", with a flattest-point fallback only when nothing satisfied the
    bound. Scored against human boxes on 200 labelled images, that is right
    only while the bound actually bites. On 1080p broadcast frames it does --
    40 persons/frame at conf 0.02 -- and the rule landed within 0.00 F1 of the
    best grid point in all four conditions tested.

    A view covering part of the pitch never reaches the bound. The Veo footage
    gives 18 persons/frame at conf 0.05, so every threshold satisfies it and
    "lowest satisfying" degenerates to the MIN_USABLE_CONF floor, chosen for
    no reason connected to the video. Reading that off the labelled curve, the
    floor is worth F1 0.80 where 0.89 was available at conf 0.20.

    So the flattest point is the rule and the roster is a constraint on it.
    Between the noise-dominated low end, where lowering the threshold adds
    detections quickly, and the high end, where it starts removing real
    players, the curve has a plateau; its flattest step is where noise is
    exhausted and players are not yet being lost.
    """
    curve = []
    for c in CONF_GRID:
        counts = [float((x >= c).sum()) for x in player_conf]
        curve.append((c, float(np.median(counts)) if counts else 0.0))
    by_conf = dict(curve)

    usable = [c for c, n in curve
              if c >= MIN_USABLE_CONF and n <= MAX_PERSONS_PER_FRAME]
    if not usable:
        # Over the bound everywhere -- a masking or crowd problem. Still
        # refuse the sub-noise end of the grid.
        usable = [c for c, _ in curve if c >= MIN_USABLE_CONF] or list(by_conf)

    best_drop, best_conf = None, None
    for i in range(len(curve) - 1):
        c, n = curve[i]
        if c not in usable:
            continue
        drop = n - curve[i + 1][1]
        # Strict improvement, scanning upward, so a tie keeps the lower
        # threshold and with it the recall.
        if best_drop is None or drop < best_drop:
            best_drop, best_conf = drop, c

    if best_conf is None:
        best_conf = usable[0]
    return best_conf, by_conf[best_conf]


def _pick_ball_conf(ball_conf: list[np.ndarray]) -> tuple[float, float]:
    """Lowest confidence keeping candidates near TARGET_BALL_PER_FRAME.

    Deliberately not the F1-optimal threshold. Measured on labelled images,
    F1 peaks near conf 0.25 (P 0.70 / R 0.33) while this rule picks 0.05
    (P 0.32 / R 0.51). Recall is what matters here: the Viterbi selection step
    filters false candidates using motion continuity, but it cannot recover a
    ball that was never detected.
    """
    best = CONF_GRID[-1], 0.0
    for c in CONF_GRID:
        counts = [float((x >= c).sum()) for x in ball_conf]
        mean = float(np.mean(counts)) if counts else 0.0
        if mean <= TARGET_BALL_PER_FRAME:
            return c, mean
        best = c, mean
    return best


def _ball_rate(ball_conf: list[np.ndarray], conf: float) -> float:
    """Fraction of probed frames with at least one ball candidate."""
    if not ball_conf:
        return 0.0
    return float(np.mean([1.0 if (x >= conf).sum() else 0.0 for x in ball_conf]))


def _measure_camera_motion(video: str, frames: list[int]) -> float:
    """Mean background movement between consecutive frames, in pixels.

    The mean, not the median, and over a longer sample than it used to be.
    A Veo camera synthesises its view from a panorama and the virtual camera
    is *bursty*: on a three-minute clip 32% of frames move under a pixel and
    the rest swing, with a per-frame median of 2.07 px and a mean of 3.22.
    A median over five short runs collapses to nearly zero whenever the runs
    happen to land in the quiet passages -- simulated on the measured motion
    it returns under a pixel 14% of the time, and it did exactly that on this
    clip, reporting 0.28 px/frame against a true 2.07.
    
    That silently turned camera compensation off for a clip whose view drifts
    1911 px, nearly three frame widths. The sampling below returns under a
    pixel in 0% of simulated draws.
    """
    from .camera_motion import (_background_mask, static_graphics_mask,
                                FEATURE_PARAMS, LK_PARAMS)

    # Read contiguous runs rather than seeking to each pair. Seeking
    # compressed video is not frame-accurate, so a "consecutive" pair can span
    # several frames and overstate the motion.
    # Burned-in graphics never move, so features on them report no motion and
    # pull this estimate toward zero -- the same fault the estimator proper
    # had. Excluding them raised the measured motion by 9% there.
    graphics = static_graphics_mask(video)

    cap = cv2.VideoCapture(video)
    steps = []
    anchors = frames[:: max(1, len(frames) // 10)][:10]
    for anchor in anchors:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(anchor))
        ok, prev = cap.read()
        if not ok:
            continue
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        for _ in range(25):
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            seed = _background_mask(prev)
            if graphics is not None:
                seed = cv2.bitwise_and(seed, graphics)
            pts = cv2.goodFeaturesToTrack(
                prev_gray, mask=seed, **FEATURE_PARAMS)
            if pts is not None and len(pts) >= 8:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, pts, None, **LK_PARAMS)
                if nxt is not None and status is not None:
                    good_new = nxt[status.flatten() == 1].reshape(-1, 2)
                    good_old = pts[status.flatten() == 1].reshape(-1, 2)
                    if len(good_new) >= 8:
                        disp = np.median(good_new - good_old, axis=0)
                        steps.append(float(np.hypot(*disp)))
            prev, prev_gray = frame, gray
    cap.release()
    return float(np.mean(steps)) if steps else 0.0


def profile_video(video: str, model_name: str = "yolov8m.pt",
                  n_samples: int = 30, verbose: bool = True) -> VideoProfile:
    """Derive detection parameters by probing the video."""
    cap = cv2.VideoCapture(video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if n_frames <= 0:
        raise ValueError(f"cannot read frames from {video}")

    frames = _sample_indices(n_frames, n_samples)
    model = YOLO(model_name)
    notes: list[str] = []

    if verbose:
        print(f"  probing {len(frames)} frames spread across "
              f"{n_frames} ({width}x{height})")

    # Input size first: it changes what is detectable at all, so the
    # confidence thresholds must be chosen at the size actually used.
    per_imgsz = {}
    for imgsz in IMGSZ_OPTIONS:
        p_conf, b_conf, heights = _probe_detections(video, model, frames, imgsz)
        rate = _ball_rate(b_conf, 0.10)
        per_imgsz[imgsz] = (p_conf, b_conf, heights, rate)
        if verbose:
            print(f"    imgsz {imgsz:>4}: ball in {rate*100:.0f}% of probed frames")

    best_rate = max(v[3] for v in per_imgsz.values())
    imgsz = next(
        (s for s in IMGSZ_OPTIONS
         if per_imgsz[s][3] >= IMGSZ_ACCEPT_FRACTION * best_rate),
        IMGSZ_OPTIONS[-1],
    )
    player_conf, ball_conf, heights, _ = per_imgsz[imgsz]

    if verbose:
        counts_by_conf = {
            c: float(np.median([float((x >= c).sum()) for x in player_conf]))
            for c in CONF_GRID
        }
        print("    persons/frame by confidence: "
              + ", ".join(f"{c:.2f}->{n:.0f}" for c, n in counts_by_conf.items()))

    conf_player, players_per_frame = _pick_player_conf(player_conf)
    conf_ball, ball_per_frame = _pick_ball_conf(ball_conf)

    if not heights:
        raise ValueError("no confident player detections; cannot establish scale")
    player_height_px = float(np.median(heights))
    px_per_m = player_height_px / PLAYER_HEIGHT_M
    ball_size_px = px_per_m * 0.22

    motion = _measure_camera_motion(video, frames)
    compensate = motion >= CAMERA_MOTION_THRESHOLD_PX

    if players_per_frame < 12:
        notes.append(
            f"only {players_per_frame:.0f} players/frame at the chosen "
            "threshold; the view may cover part of the pitch, or detection "
            "is struggling")
    if ball_size_px < 5:
        notes.append(
            f"ball is only ~{ball_size_px:.1f} px across; detection will be "
            "unreliable regardless of threshold")
    if best_rate < 0.5:
        notes.append(
            f"ball found in only {best_rate*100:.0f}% of probed frames at "
            "best; event detection will be limited by ball coverage")

    profile = VideoProfile(
        video=video, width=width, height=height, fps=fps, n_frames=n_frames,
        imgsz=imgsz, conf_player=conf_player, conf_ball=conf_ball,
        px_per_m=px_per_m, player_height_px=player_height_px,
        ball_size_px=ball_size_px,
        min_detection_height_px=max(8.0, player_height_px * MIN_HEIGHT_FRACTION),
        compensate_camera=compensate, camera_motion_px=motion,
        players_per_frame=players_per_frame, ball_per_frame=ball_per_frame,
        notes=notes,
    )
    if verbose:
        print(profile.summary())
    return profile
