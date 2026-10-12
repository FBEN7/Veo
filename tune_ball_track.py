"""Protocol v4-r1: the ball tracker's settings (`src/ball_track.py`), fixed
by the protocol -- nothing is tuned -- and frozen into track_params.json
with its `<out>.sha256` before the tracker runs on U-test.

    python tune_ball_track.py --out <scratchpad>/ball/track_params.json

The record is {"frozen": true, "params": PARAMS}: tau 0.25 (the U
calibration's), top_k 4, c_null 1, c_static 2, gap 6 frames,
max_fill_speed 35 m/s (the linker's free speed) and the fallback on, which
is `ball_track.DEFAULTS` with tau (refused if the two ever differ).
`eval_ball_detector.py --track-params <out>` reads it only when it says
"frozen": true and `<out>.sha256` matches it. An existing file is left as
it is when it holds this very record (its .sha256 checked, or written when
missing); a file with anything else in it, or a .sha256 that does not
match, is refused, never replaced.

The draft's grid search on U-E (c_null, c_static, gap and the fallback,
chosen by FF-auto under a placebo constraint) is withdrawn by v4-r1: it
tuned exactly what FF-auto cannot judge. This script reads no data.
`--check` runs the self-check (a temporary directory only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# Protocol v4-r1's frozen settings, written out so a change to
# ball_track.DEFAULTS cannot slip into the frozen file unseen.
PARAMS = {"tau": 0.25, "top_k": 4, "c_null": 1.0, "c_static": 2.0, "gap": 6,
          "max_fill_speed": 35.0, "fallback": True}


def record() -> dict:
    """The frozen record; refuses when ball_track.DEFAULTS is not PARAMS
    less tau."""
    from src import ball_track as bt

    if {k: v for k, v in PARAMS.items() if k != "tau"} != bt.DEFAULTS:
        raise ValueError(f"ball_track.DEFAULTS {bt.DEFAULTS} are not protocol v4-r1's "
                         f"{PARAMS}")
    return {"frozen": True, "params": dict(PARAMS)}


def text() -> bytes:
    return (json.dumps(record(), indent=1) + "\n").encode()


def freeze(out) -> str:
    """Writes `out` and `<out>.sha256` (or checks them when there); returns
    the sha256."""
    out = Path(out)
    digest_path = Path(f"{out}.sha256")
    want = text()
    digest = hashlib.sha256(want).hexdigest()
    if out.exists():
        if out.read_bytes() != want:
            raise ValueError(f"{out} exists with another record; a frozen file is not "
                             f"replaced")
    else:
        if digest_path.exists():
            raise ValueError(f"{digest_path} exists without {out.name}")
        out.write_bytes(want)
    line = f"{digest}  {out.name}\n"
    if digest_path.exists():
        got = (digest_path.read_text().split() or [""])[0]
        if got != digest:
            raise ValueError(f"{digest_path} says {got}, not {out.name}'s {digest}")
    else:
        digest_path.write_text(line)
    return digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="track_params.json (its .sha256 next to it)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        _check()
        return
    if not args.out:
        ap.error("--out is needed")
    try:
        digest = freeze(args.out)
    except ValueError as e:
        ap.error(str(e))
    print(f"  {args.out}: frozen {PARAMS} (sha256 {digest})")


def _check():
    """The record is PARAMS = ball_track.DEFAULTS with tau, written once
    with its sha256, read back by `eval_ball_detector.read_track_params`
    (and as a --track-params spec); writing again is a no-op; another
    record, a stale .sha256 or a lone .sha256 are refused."""
    import tempfile

    import eval_ball_detector as ev

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "track_params.json"
        d = freeze(out)
        assert d == hashlib.sha256(out.read_bytes()).hexdigest()
        assert Path(f"{out}.sha256").read_text() == f"{d}  {out.name}\n"
        got = ev.read_track_params(str(out))
        assert got == PARAMS and {**ev.TRACK_DEFAULT, **got} == PARAMS
        name, path, params, given = ev.track_specs(["T=x_{window}.parquet"], [str(out)])[0]
        assert params == PARAMS and given == [{"path": str(out.resolve()), "sha256": d}]
        assert freeze(out) == d and out.read_bytes() == text()
        for bad in (lambda: (out.write_text(json.dumps({"frozen": True, "params": {
                        **PARAMS, "gap": 12}})), freeze(out)),
                    lambda: (out.write_bytes(text()), Path(f"{out}.sha256").write_text(
                        "0" * 64 + "\n"), freeze(out)),
                    lambda: (out.unlink(), freeze(out))):
            try:
                bad()
                raise AssertionError("another record or a stale sha256 was accepted")
            except ValueError:
                pass
    print("  tune_ball_track: the frozen v4-r1 settings, their sha256, read back by "
          "eval_ball_detector, no silent replacement: ok")


if __name__ == "__main__":
    main()
