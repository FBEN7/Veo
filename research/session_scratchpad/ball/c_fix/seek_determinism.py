"""A resumed stream run seeks (stream_frames with first > 0): are the frames
after a seek those of a sequential decode? On the training window
st2_118575_1st_f015000.mp4 (x264), sequential 0..1110 vs seeks to 600 and
1000 (beyond the first keyframes)."""
import hashlib
import sys

sys.path.insert(0, "/home/user/Veo")
import av  # noqa: E402

import build_ball_strips as bs  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402

path = str(DATA_DIR / "soccertrack_v2" / "st2_118575_1st_f015000.mp4")
with av.open(path) as c:
    s = c.streams.video[0]
    keys = []
    for p in c.demux(s):
        if p.pts is not None and p.is_keyframe:
            keys.append(int(round(float(p.pts * s.time_base) * 25)))
        if p.pts is not None and float(p.pts * s.time_base) * 25 > 1200:
            break
print("keyframes up to 1200:", sorted(keys))
want = set(range(595, 640)) | set(range(1000, 1111))
with av.open(path) as c:
    seq = {k: hashlib.md5(f().tobytes()).hexdigest()
           for k, f in bs.stream_frames(c, 0, 1110) if k in want}
for first, last in ((600, 639), (1000, 1110)):
    with av.open(path) as c:
        got = {k: hashlib.md5(f().tobytes()).hexdigest()
               for k, f in bs.stream_frames(c, first, last)}
    same = all(got[k] == seq[k] for k in range(first, last + 1))
    print(f"seek to {first}: frames {min(got)}-{max(got)} ({len(got)}), identical to "
          f"the sequential decode: {same}")
    assert same and sorted(got) == list(range(first, last + 1))
print("OK")
