"""SoccerTrack v2: whole-pitch fixed panoramic footage with ground truth.

Ten university matches (934 minutes) filmed by fixed panoramic camera
systems that keep the whole pitch in view -- no cuts, no replays, the
footage a Veo panorama gives -- released with, per frame, every player's
pitch position, shirt number, persistent id, role and side ("GSR"), and
21,428 ball actions with the player who made them ("BAS"). CC BY 4.0,
pseudonymised (numbers and integer ids, no names). Scott et al.,
arXiv 2508.01802; https://huggingface.co/datasets/atomscott/soccertrack-v2.

The dataset is gated: a Hugging Face token (`HF_TOKEN` or `HF_SECRET`, or
a proxy that adds it) of an account that accepted its terms is needed. Nothing is downloaded whole -- a half's GSR
file is ~2.7 GB of JSON and its video a 45-minute 4K panorama, against a
few GB of disk here -- so:

- **GSR** is streamed and its object records written once to a compact
  table (`gsr_table`, one parquet per half in `.cache/soccertrack_v2`);
- **BAS** files are small and read whole (`events`);
- **video** is read by HTTP range requests (`RemoteFile`), so the decoder
  seeks straight to the minutes wanted (`read_frames`, `cut_clip`).

The released files differ from the repository's own format docs, checked
against its loader and notes (github.com/AtomScott/SoccerTrack-v2,
`src/data_utils/soccertrack_v2.py`, `docs/format-*.md`), and these are
what is read here:

- GSR is SoccerNet `Labels-GameState.json`: `info`, `images`,
  `annotations`, `categories`; `image_id` is a string, a sequence prefix
  and a 1-indexed six-digit frame ("3000001" is frame 0); role, `jersey`
  (a string), `team` and `player_id` sit under `attributes`; the pitch
  position is `bbox_pitch.x_bottom_middle` / `y_bottom_middle` in metres,
  origin at the centre spot; `bbox_image` is a dict in true pixels (the
  declared image size, 3840 x 1504, is wrong for the 4096-wide matches);
- BAS events are under `actions` (or `annotations`), labels in capitals.
  From release v1.2 each carries `frame`, its video and GSR frame
  (1-based, like the GSR image ids), and `position` is milliseconds from
  the half video's first decoded frame; earlier
  releases had `position` from the start of the *match*, so a second-half
  event is placed by subtracting 45 minutes (`_event_frame`).

The direction of the pitch `y` axis is stated both ways (the dataset's
README: towards the touchline nearer the camera; the format doc: towards
the far one); `y_towards_camera` settles it from the data: on M2 the
README is right (y and the box bottom correlate at 0.93).

Measured on the real files (v1.2): frames are 4096 x 1080 on M2 (the
`images` sizes are right); the GSR holds only players and goalkeepers (no
referees, no ball); every image box is 42-43 px tall and 17-81 px wide,
wider than the players (~18 px), and sits within ~10 px (median) of the
detected feet with no time offset; BAS `frame` is 1-based.

    python -m src.soccertrack_v2          # self-check on synthetic files
"""

from __future__ import annotations

import io
import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "atomscott/soccertrack-v2"
REVISION = "v1.2"
FPS = 25
HALF_MS = 45 * 60 * 1000
# Match id -> (the paper's label, the SoccerTrack Challenge 2025 split).
MATCHES = {
    "117092": ("M1", "train"), "117093": ("M2", "validation"),
    "118575": ("M3", "train"), "118576": ("M4", "train"),
    "118577": ("M5", "train"), "118578": ("M6", "train"),
    "128057": ("M7", "test"), "128058": ("M8", "train"),
    "132831": ("M9", "test"), "132877": ("M10", "validation"),
}
# BAS label -> the type in this project's event format (src/event_schema.py),
# None where it has no counterpart (the label is kept either way). Restarts
# by hand or foot are passes in SPADL; a header, a block or a free kick may
# be a pass, a shot or a clearance, which the label does not say.
KINDS = {
    "PASS": "pass", "HIGH PASS": "pass", "CROSS": "pass", "THROW IN": "pass",
    "DRIVE": "carry", "SHOT": "shot", "GOAL": "goal",
    "PLAYER SUCCESSFUL TACKLE": "tackle", "OUT": "out",
    "HEADER": None, "BALL PLAYER BLOCK": None, "FREE KICK": None,
}
GSR_COLUMNS = ["frame", "track_id", "player_id", "role", "jersey", "side",
               "x", "y", "bx", "by", "bw", "bh"]


def cache_dir() -> Path:
    from .paths import CACHE_DIR

    return Path(CACHE_DIR) / "soccertrack_v2"


# ---- Access -------------------------------------------------------------------

def token() -> str | None:
    """The Hugging Face token from the environment (HF_TOKEN, HF_SECRET or
    HUGGING_FACE_HUB_TOKEN), or None when an outbound proxy authenticates
    requests to huggingface.co itself (checked with `whoami`)."""
    for k in ("HF_TOKEN", "HF_SECRET", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k):
            return os.environ[k]
    import requests

    r = requests.get("https://huggingface.co/api/whoami-v2", timeout=30)
    if r.ok:
        return None
    raise RuntimeError(
        "SoccerTrack v2 is gated: set HF_TOKEN to a read token of a "
        "Hugging Face account that accepted the dataset's terms at "
        f"https://huggingface.co/datasets/{REPO}")


def auth_headers() -> dict:
    t = token()
    return {"Authorization": f"Bearer {t}"} if t else {}


def url(path: str, revision: str = REVISION) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{path}"


def _session():
    import requests

    s = requests.Session()
    s.headers.update(auth_headers())
    return s


def signed_url(path: str, revision: str = REVISION) -> str:
    """Where the file is actually served from: Hugging Face answers the
    authenticated request with a redirect to a signed address that needs no
    token, which the range requests then use."""
    r = _session().head(url(path, revision), allow_redirects=False, timeout=60)
    if r.status_code in (301, 302, 303, 307, 308):
        return r.headers["Location"]
    r.raise_for_status()
    return url(path, revision)


def open_stream(path: str, revision: str = REVISION):
    """A file of the dataset as a byte stream, without storing it."""
    r = _session().get(url(path, revision), stream=True, timeout=60)
    r.raise_for_status()
    r.raw.decode_content = True
    return r.raw


class RemoteFile(io.RawIOBase):
    """A remote file, read by HTTP range requests in `block`-sized pieces
    (the last `keep` kept), seekable -- so a video decoder can jump to the
    minute wanted without the whole file being fetched. `fetch(start, end)`
    returns bytes start..end inclusive; by default from `address`."""

    def __init__(self, address: str | None = None, size: int | None = None,
                 fetch=None, block: int = 4 << 20, keep: int = 32):
        super().__init__()
        if fetch is None:
            import requests

            s = requests.Session()

            def fetch(a, b):
                r = s.get(address, headers={"Range": f"bytes={a}-{b}"},
                          timeout=120)
                r.raise_for_status()
                return r.content

            if size is None:
                r = s.head(address, allow_redirects=True, timeout=60)
                r.raise_for_status()
                size = int(r.headers["Content-Length"])
        if size is None:
            raise ValueError("size is needed with a custom fetch")
        self._fetch, self.size = fetch, size
        self._block, self._keep = block, keep
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        self._pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos,
                io.SEEK_END: self.size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def _piece(self, k: int) -> bytes:
        if k in self._blocks:
            self._blocks.move_to_end(k)
            return self._blocks[k]
        a = k * self._block
        data = self._fetch(a, min(a + self._block, self.size) - 1)
        self._blocks[k] = data
        while len(self._blocks) > self._keep:
            self._blocks.popitem(last=False)
        return data

    def readinto(self, buf) -> int:
        n = min(len(buf), self.size - self._pos)
        if n <= 0:
            return 0
        done = 0
        while done < n:
            k, off = divmod(self._pos + done, self._block)
            piece = self._piece(k)[off:off + n - done]
            buf[done:done + len(piece)] = piece
            done += len(piece)
        self._pos += n
        return n


# ---- GSR ----------------------------------------------------------------------

def iter_array(stream, key: str, chunk: int = 1 << 20):
    """The items of the top-level array `key` of a JSON object read from a
    byte stream, one at a time, without holding the document: a half's GSR
    file is ~2.7 GB."""
    dec = json.JSONDecoder()
    buf, eof = "", False

    def more():
        nonlocal buf, eof
        data = stream.read(chunk)
        if not data:
            eof = True
        else:
            buf += data.decode("utf-8") if isinstance(data, bytes) else data

    marker = f'"{key}"'
    # Find `"key"`, then its colon and opening bracket.
    while True:
        i = buf.find(marker)
        if i >= 0:
            j = buf.find("[", i + len(marker))
            if j >= 0 and buf[i + len(marker):j].strip() == ":":
                buf = buf[j + 1:]
                break
            if j >= 0:
                buf = buf[i + len(marker):]
                continue
        if eof:
            return
        if i < 0:
            buf = buf[-len(marker):]
        more()
    while True:
        k = 0
        while k < len(buf) and buf[k] in " \t\r\n,":
            k += 1
        buf = buf[k:]
        if buf.startswith("]"):
            return
        try:
            item, end = dec.raw_decode(buf)
        except json.JSONDecodeError:
            if eof:
                raise
            more()
            continue
        yield item
        buf = buf[end:]
        if len(buf) < chunk // 4 and not eof:
            more()


def frame_of(image_id) -> int:
    """GSR `image_id` ("3000001": a sequence prefix and a 1-indexed six-digit
    frame) to the 0-indexed video frame."""
    return int(str(image_id)[-6:]) - 1


def gsr_row(rec: dict) -> dict | None:
    """One GSR object record as a flat row; None for pitch and camera
    records."""
    if rec.get("supercategory", "object") != "object" or "attributes" not in rec:
        return None
    att = rec["attributes"] or {}
    pitch = rec.get("bbox_pitch") or {}
    box = rec.get("bbox_image") or {}
    jersey = att.get("jersey")
    return {
        "frame": frame_of(rec["image_id"]),
        "track_id": int(rec["track_id"]),
        "player_id": None if att.get("player_id") is None else str(att["player_id"]),
        "role": att.get("role"),
        "jersey": int(jersey) if jersey not in (None, "") else None,
        "side": att.get("team"),
        "x": pitch.get("x_bottom_middle"), "y": pitch.get("y_bottom_middle"),
        "bx": box.get("x"), "by": box.get("y"),
        "bw": box.get("w"), "bh": box.get("h"),
    }


def gsr_rows(stream) -> pd.DataFrame:
    rows = [r for r in map(gsr_row, iter_array(stream, "annotations")) if r]
    df = pd.DataFrame(rows, columns=GSR_COLUMNS)
    df["jersey"] = df["jersey"].astype("Int64")
    return df


def gsr_table(match: str, half: int, refresh: bool = False) -> pd.DataFrame:
    """Every object record of a half: frame, track_id, player_id, role,
    jersey, side, pitch x / y (metres) and image box bx, by, bw, bh (pixels);
    streamed from the dataset once and kept as parquet."""
    path = cache_dir() / "gsr" / f"{match}_{_half(half)}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = gsr_rows(open_stream(f"gsr/{match}/{match}_{_half(half)}.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


SIDE_LINES = ("Side line top", "Side line right", "Side line bottom",
              "Side line left")


def outline_of(lines: dict, width: int, height: int) -> list[list[float]]:
    """The pitch outline in pixels from a GSR pitch record's `lines`
    (points normalised by the image size): the four side lines chained end
    to end, each turned round where needed."""
    segs = [[(p["x"] * width, p["y"] * height) for p in lines[k]
             if isinstance(p, dict)] for k in SIDE_LINES]
    out = list(segs[0])
    rest = segs[1:]
    while rest:
        end = np.array(out[-1])
        k = min(range(len(rest)), key=lambda i: min(
            np.hypot(*(end - rest[i][0])), np.hypot(*(end - rest[i][-1]))))
        seg = rest.pop(k)
        if np.hypot(*(end - seg[-1])) < np.hypot(*(end - seg[0])):
            seg = seg[::-1]
        out += seg[1:]
    return [[round(x, 1), round(y, 1)] for x, y in out]


def pitch_outline(match: str, half: int, width: int,
                  height: int) -> list[list[float]]:
    """The pitch outline of a half's fixed view, from the first GSR pitch
    record with all four side lines (read from the start of the file; the
    camera does not move)."""
    for rec in iter_array(open_stream(f"gsr/{match}/{match}_{_half(half)}.json"),
                          "annotations"):
        lines = rec.get("lines") or {}
        if rec.get("supercategory") == "pitch" and all(
                sum(isinstance(p, dict) for p in lines.get(k, [])) >= 2
                for k in SIDE_LINES):
            return outline_of(lines, width, height)
    raise ValueError(f"no pitch record with side lines in {match} {half}")


# ---- Calibration ----------------------------------------------------------------

def calibration(match: str) -> dict:
    """The release's camera model of a match (`raw/<match>/`): fisheye
    intrinsics K, D, the undistorted camera Knew, and the inverse of the
    homography from pitch metres (corner origin, 105 x 68) to undistorted
    pixels. Kept in the cache; the intrinsics file's pickled rotation and
    translation are not read."""
    root = cache_dir() / "calib"
    root.mkdir(parents=True, exist_ok=True)
    for name in ("camera_intrinsics.npz", "homography.npy"):
        path = root / f"{match}_{name}"
        if not path.exists():
            path.write_bytes(open_stream(f"raw/{match}/{match}_{name}").read())
    z = np.load(root / f"{match}_camera_intrinsics.npz", allow_pickle=False)
    H = np.load(root / f"{match}_homography.npy", allow_pickle=False)
    return {"K": z["K"], "D": z["D"], "Knew": z["Knew"], "Hinv": np.linalg.inv(H)}


def image_to_pitch(uv, cal: dict) -> np.ndarray:
    """(N, 2) pixels of a match's panorama to (N, 2) pitch metres in the
    dataset's convention (origin the centre spot, x to the right as seen
    from the camera, y towards it); NaN at or above the horizon. The
    points are undistorted with the fisheye model, then mapped through the
    inverse homography. On M2 the ground-truth boxes' feet land a median
    0.38 m from their pitch positions (the positions are on a 1.05 x
    0.68 m grid, worth 0.34 m alone)."""
    import cv2

    # cv2 misreads strided views without complaint: always a fresh array.
    uv = np.ascontiguousarray(np.asarray(uv, dtype=np.float64)
                              .reshape(-1, 1, 2))
    und = cv2.fisheye.undistortPoints(uv, cal["K"], cal["D"],
                                      P=cal["Knew"]).reshape(-1, 2)
    q = np.c_[und, np.ones(len(und))] @ cal["Hinv"].T
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = q[:, :2] / q[:, 2:] - (52.5, 34.0)
    xy[q[:, 2] <= 0] = np.nan
    return xy


def y_towards_camera(gsr: pd.DataFrame) -> bool:
    """Whether pitch y grows towards the camera: the camera looks down on
    the pitch, so feet nearer it are lower in the picture, and the box
    bottom (by + bh) rises with y if it does. (Box height would say the same
    in principle, but the released boxes are all 42-43 px tall.)"""
    p = gsr[(gsr.role == "player") & gsr.by.notna() & gsr.y.notna()]
    foot = p.by.astype(float) + p.bh.astype(float)
    return bool(np.corrcoef(p.y.astype(float), foot)[0, 1] > 0)


def _half(half: int) -> str:
    return {1: "1st", 2: "2nd"}[half]


# ---- BAS ----------------------------------------------------------------------

def _event_frame(a: dict, half: int) -> int:
    """The 0-indexed half-video frame of an event: its `frame` (release v1.2
    on, 1-based), else from `position`, which before v1.2 counted from the
    start of the match."""
    if a.get("frame") is not None:
        return int(a["frame"]) - 1
    ms = int(a["position"])
    if half == 2 and ms >= HALF_MS:
        ms -= HALF_MS
    return int(round(ms * FPS / 1000))


def parse_events(blob: dict) -> pd.DataFrame:
    """One match's BAS file as a table: half, frame, label (capitals), kind
    (this project's type, or None), side, player_id, visibility."""
    events = next(blob[k] for k in ("actions", "annotations") if k in blob)
    rows = []
    for a in events:
        gt = str(a["gameTime"])
        if " - " in gt:
            half = int(gt.split(" - ", 1)[0])
        else:   # a third period some matches played; not filmed
            continue
        label = str(a["label"]).upper()
        if label not in KINDS:
            raise ValueError(f"unknown BAS label {a['label']!r}")
        rows.append({"half": half, "frame": _event_frame(a, half),
                     "label": label, "kind": KINDS[label],
                     "side": a.get("team"),
                     "player_id": None if a.get("player_id") is None
                     else str(a["player_id"]),
                     "visibility": a.get("visibility")})
    return pd.DataFrame(rows, columns=["half", "frame", "label", "kind",
                                       "side", "player_id", "visibility"])


def events(match: str) -> pd.DataFrame:
    path = cache_dir() / "bas" / f"{match}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(open_stream(
            f"bas/{match}/{match}_12_class_events.json").read())
    return parse_events(json.loads(path.read_text()))


def with_actors(ev: pd.DataFrame, gsr: pd.DataFrame) -> pd.DataFrame:
    """Events of one half with their actor's shirt number and pitch position
    on the event's frame (from the GSR record with the same player_id)."""
    g = gsr[gsr.player_id.notna()][["frame", "player_id", "jersey", "x", "y"]]
    return ev.merge(g, on=["frame", "player_id"], how="left")


# ---- Video --------------------------------------------------------------------

def video_path(match: str, half: int) -> str:
    return f"videos/{match}/{match}_panorama_{_half(half)}_half.mp4"


def open_video(match: str, half: int):
    """A half's panorama, read on demand (PyAV container)."""
    import av

    return av.open(RemoteFile(signed_url(video_path(match, half))))


def read_frames(container, start: int, count: int, step: int = 1):
    """(frame index, BGR image) for frames start, start + step, ... of the
    video, `count` of them, seeking to the keyframe before `start`."""
    stream = container.streams.video[0]
    tb = float(stream.time_base)
    t0 = float(stream.start_time or 0) * tb
    container.seek(int((start / FPS + t0) / tb), stream=stream,
                   backward=True, any_frame=False)
    want = set(range(start, start + count * step, step))
    for frame in container.decode(stream):
        k = int(round((float(frame.pts) * tb - t0) * FPS))
        if k in want:
            yield k, frame.to_ndarray(format="bgr24")
            want.discard(k)
            if not want:
                return
        elif k > max(want):
            return


def write_clip(container, start: int, count: int, out: Path,
               crf: int = 18) -> dict:
    """Write frames start .. start + count - 1 of a video (PyAV container)
    as H.264 at full resolution -- plays in a browser, so labelling pages
    can use it -- and return what was written."""
    import av

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dst = av.open(str(out), "w")
    stream = None
    n = 0
    for k, image in read_frames(container, start, count):
        if stream is None:
            stream = dst.add_stream("libx264", rate=FPS)
            stream.width, stream.height = image.shape[1], image.shape[0]
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": str(crf), "preset": "veryfast"}
        for p in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
            dst.mux(p)
        n += 1
    if stream is not None:
        for p in stream.encode():
            dst.mux(p)
    dst.close()
    return {"start_frame": start, "frames": n, "fps": FPS, "path": str(out),
            "width": stream.width if stream else None,
            "height": stream.height if stream else None}


def cut_clip(match: str, half: int, start: int, count: int, out: Path,
             crf: int = 18) -> dict:
    """Frames start .. start + count - 1 of a half's panorama, fetched by
    range requests, written as H.264 at full resolution."""
    src = open_video(match, half)
    try:
        info = write_clip(src, start, count, out, crf)
    finally:
        src.close()
    return {"match": match, "half": half, **info}


# ---- Self-check ---------------------------------------------------------------

def _check():
    # GSR in the shipped form, streamed through a small chunk size so items
    # straddle reads; pitch and camera records are skipped.
    gsr = {
        "info": {"version": "1.3", "seq_length": 3, "frame_rate": 25},
        "images": [{"image_id": "3000001", "width": 3840, "height": 1504}],
        "annotations": [
            {"id": "1", "image_id": "3000001", "track_id": 1,
             "supercategory": "object", "category_id": 1,
             "attributes": {"role": "player", "jersey": "12", "team": "left",
                            "player_id": 170959},
             "bbox_image": {"x": 1517, "y": 152, "w": 58, "h": 43,
                            "x_center": 1546.0, "y_center": 173.5},
             "bbox_pitch": {"x_bottom_middle": -23.1, "y_bottom_middle": -21.76,
                            "x_bottom_left": -23.1, "y_bottom_left": -21.76,
                            "x_bottom_right": -23.1, "y_bottom_right": -21.76}},
            {"id": "2", "image_id": "3000001", "supercategory": "pitch",
             "lines": {"Side line top": [{"x": 0.1, "y": 0.2}]}},
            {"id": "3", "image_id": "3000003", "track_id": 2,
             "supercategory": "object", "category_id": 3,
             "attributes": {"role": "referee", "jersey": None, "team": None,
                            "player_id": None},
             "bbox_image": {"x": 10, "y": 900, "w": 60, "h": 140},
             "bbox_pitch": {"x_bottom_middle": 0.5, "y_bottom_middle": 30.0}},
        ],
        "categories": [{"id": 1, "name": "player"}],
    }
    raw = json.dumps(gsr).encode()
    df = gsr_rows(_Chunks(raw, 37))
    assert list(df.frame) == [0, 2], df
    assert df.jersey.iloc[0] == 12 and pd.isna(df.jersey.iloc[1])
    assert df.player_id.iloc[0] == "170959" and df.side.iloc[0] == "left"
    assert (df.x.iloc[0], df.y.iloc[0]) == (-23.1, -21.76)
    assert (df.bw.iloc[1], df.bh.iloc[1]) == (60, 140)
    assert y_towards_camera(pd.DataFrame(
        {"role": ["player"] * 3, "y": [-30.0, 0.0, 30.0],
         "by": [150.0, 400.0, 700.0], "bh": [43.0, 43.0, 43.0]}))
    # The key named inside an earlier value, and nested arrays, are not
    # taken for the array itself.
    tricky = json.dumps({"info": {"note": "annotations", "x": [1, 2]},
                         "annotations": [{"a": [1, {"b": "]"}]}, {"a": 2}]})
    assert list(iter_array(_Chunks(tricky.encode(), 5), "annotations")) == \
        [{"a": [1, {"b": "]"}]}, {"a": 2}]
    # BAS: v1.2 carries the frame; before, position counted from the match
    # start, and a second-half event is placed 45 minutes earlier.
    v12 = {"actions": [
        {"gameTime": "1 - 00:02", "position": "2000", "frame": 50,
         "label": "PASS", "team": "left", "player_id": 170959},
        {"gameTime": "2 - 00:04", "position": "4000", "frame": 100,
         "label": "HIGH PASS", "team": "right"},
        {"gameTime": "90:01", "position": "5401000", "label": "SHOT"}]}
    ev = parse_events(v12)
    assert list(ev.frame) == [49, 99] and list(ev.kind) == ["pass", "pass"]
    assert ev.player_id.iloc[0] == "170959" and pd.isna(ev.player_id.iloc[1])
    v11 = {"annotations": [
        {"gameTime": "2 - 00:04", "position": str(HALF_MS + 4000),
         "label": "Player Successful Tackle", "team": "left"}]}
    ev = parse_events(v11)
    assert ev.frame.iloc[0] == 100 and ev.kind.iloc[0] == "tackle"
    try:
        parse_events({"actions": [{"gameTime": "1 - 00:00", "position": "0",
                                   "label": "PENALTY"}]})
        raise AssertionError("an unknown label was accepted")
    except ValueError:
        pass
    # Actors take their shirt and place from the GSR record of that frame.
    acted = with_actors(pd.DataFrame({"frame": [0], "player_id": ["170959"]}), df)
    assert acted.jersey.iloc[0] == 12 and acted.x.iloc[0] == -23.1
    # The outline chains the side lines whichever way each was drawn.
    pt = lambda x, y: {"x": x, "y": y}
    box = outline_of({"Side line top": [pt(0.1, 0.1), pt(0.9, 0.1)],
                      "Side line right": [pt(0.9, 0.9), pt(0.9, 0.1)],
                      "Side line bottom": [pt(0.1, 0.9), pt(0.9, 0.9)],
                      "Side line left": [pt(0.1, 0.9), pt(0.1, 0.1)]},
                     100, 10)
    assert box == [[10, 1], [90, 1], [90, 9], [10, 9], [10, 1]], box
    # Player feet outside the outline by more than the margin are dropped,
    # balls kept.
    from .player_filter import inside_pitch

    rows = pd.DataFrame({"cls": ["player", "player", "ball", "player"],
                         "px": [50.0, 150.0, 150.0, 105.0],
                         "py": [5.0, 5.0, 5.0, 5.0]})
    kept = inside_pitch(rows, [[0, 0], [100, 0], [100, 10], [0, 10]], 10)
    assert list(kept.index) == [0, 2, 3], kept
    # Image to pitch inverts a known camera: pitch points through a
    # homography and fisheye distortion come back where they started.
    import cv2

    K = np.array([[1500.0, 0, 2000], [0, 1500, 500], [0, 0, 1]])
    D = np.array([[-0.06], [0.03], [0.0], [0.0]])
    Knew = np.array([[500.0, 0, 2000], [0, 500, 500], [0, 0, 1]])
    H = np.array([[30.0, 4, 300], [0.5, 8, 100], [0.0, 0.003, 1]])
    pts = np.array([[10.0, 5], [52.5, 34], [100, 60], [80, 20]])
    und = np.c_[pts, np.ones(4)] @ H.T
    und = und[:, :2] / und[:, 2:]
    norm = (und - Knew[:2, 2]) / np.diag(Knew)[:2]
    img = cv2.fisheye.distortPoints(norm.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    back = image_to_pitch(img, {"K": K, "D": D, "Knew": Knew,
                                "Hinv": np.linalg.inv(H)})
    assert np.allclose(back, pts - (52.5, 34.0), atol=1e-3), back
    # Range reads: across block edges, after seeks, through a small cache.
    blob = bytes(range(256)) * 41
    calls = []

    def fetch(a, b):
        calls.append((a, b))
        return blob[a:b + 1]

    f = io.BufferedReader(RemoteFile(fetch=fetch, size=len(blob), block=100,
                                     keep=2), buffer_size=64)
    f.seek(95)
    assert f.read(30) == blob[95:125]
    f.seek(-10, io.SEEK_END)
    assert f.read() == blob[-10:]
    f.seek(0)
    assert f.read() == blob
    assert all(b - a < 100 for a, b in calls)
    print("  soccertrack_v2: GSR streamed and flattened, BAS frames for both "
          "releases, actors joined, pitch outline chained and applied, "
          "image to pitch through a known camera, range reads exact")


class _Chunks:
    """A byte stream that hands out `n` bytes at a time."""

    def __init__(self, data: bytes, n: int):
        self.data, self.n, self.pos = data, n, 0

    def read(self, size: int = -1) -> bytes:
        out = self.data[self.pos:self.pos + min(self.n, size if size > 0 else self.n)]
        self.pos += len(out)
        return out


if __name__ == "__main__":
    _check()
