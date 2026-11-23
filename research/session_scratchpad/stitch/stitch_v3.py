"""Stitching v3 (plan_v3.json): v2, filling only slots that are almost
surely players.

    VEO_UNSEAL=... python stitch_v3.py
"""
import json

import numpy as np

import stitch as S
import stitch_v2 as V2

MIN_ROWS_S = 4.0
_fill = V2.fill_rows


def fill_rows(rows, frames, slot, team):
    sure = (team in ("team_A", "team_B") and (rows.frame.max() - rows.frame.min()) / S.FPS >= MIN_ROWS_S
            and abs(np.median(rows.x)) <= 52.5 and abs(np.median(rows.y)) <= 34)
    if sure:
        return _fill(rows, frames, slot, team)
    return rows.assign(kind="det", slot=slot)


V2.fill_rows = fill_rows

if __name__ == "__main__":
    import pathlib
    V2.main()
    src = S.HERE / "stitch_v2.json"
    dst = S.HERE / "stitch_v3.json"
    o = json.loads(src.read_text())
    o["plan_v3 sha256"] = (S.HERE / "plan_v3.json.sha256").read_text().split()[0]
    dst.write_text(json.dumps(o, indent=1))
