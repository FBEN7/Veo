"""Fix 1 on real data: the training window st2_118575_1st_f015000.mp4 served
through build_ball_strips._FlakyServer (signatures expire mid-decode -> 403,
then a timeout) and read with resigning_fetch + st.RemoteFile + stream_frames,
against a direct PyAV decode of the same file: identical frames. Twice: from
frame 0, and after a seek (stream_frames with first > 0)."""
import hashlib
import sys
import time

sys.path.insert(0, "/home/user/Veo")
import av  # noqa: E402

import build_ball_strips as bs  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402
from src.paths import DATA_DIR  # noqa: E402

path = DATA_DIR / "soccertrack_v2" / "st2_118575_1st_f015000.mp4"


def digests(container, first, last):
    return [(k, hashlib.md5(f().tobytes()).hexdigest())
            for k, f in bs.stream_frames(container, first, last)]


for first, last in ((0, 149), (3000, 3060)):
    t0 = time.time()
    with av.open(str(path)) as c:
        ref = digests(c, first, last)
    server = bs._FlakyServer(path)
    waits = []
    fetch, size = bs.resigning_fetch(server.sign, session=server, sleep=waits.append,
                                     log=print)
    rf = st.RemoteFile(fetch=fetch, size=size, block=256 << 10)
    c = av.open(rf)
    n_open = server.n
    # Failures after the container is open, i.e. during the decode.
    server.expire_at, server.timeout_at = n_open + 4, n_open + 9
    got = digests(c, first, last)
    c.close()
    same = got == ref
    print(f"frames {first}-{last}: {len(got)} decoded through the flaky server "
          f"({server.n} GETs, {n_open} to open; 403 at GET {server.expire_at}, "
          f"timeout at GET {server.timeout_at}); signatures {server.signs}, "
          f"retries {fetch.stats['retries']}, waits {waits}; "
          f"identical to the direct decode ({len(ref)} frames): {same}; "
          f"{time.time() - t0:.0f} s")
    assert same and len(got) == last - first + 1
    assert server.n > server.timeout_at and server.signs == 2
    assert fetch.stats["retries"] == 2 and waits == [bs.BACKOFF_S]
print("OK")
