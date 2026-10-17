"""Frames of a video by index, read forward rather than sought.

Seeking an H.264 file decodes from the previous keyframe every time: on the
SoccerNet windows `cap.set(CAP_PROP_POS_FRAMES)` cost 0.31 s a call, and
the goal placer's 230 calls on one clip took 72 s of its 191. Reading
forward and decoding only the frames asked for costs a few milliseconds a
frame skipped. Measured on reading_0737, the frames are identical either
way.
"""

from __future__ import annotations

# Further ahead than this, a seek is cheaper than grabbing every frame.
MAX_SKIP = 250


def frames(video_path: str, indices):
    """Yield (index, frame) for each wanted index that can be read, in
    increasing order."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    position = 0                 # index of the next frame `read` returns
    try:
        for index in sorted({int(i) for i in indices}):
            if index < position or index - position > MAX_SKIP:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                position = index
            while position < index:
                if not cap.grab():
                    return
                position += 1
            ok, frame = cap.read()
            position += 1
            if ok:
                yield index, frame
    finally:
        cap.release()
