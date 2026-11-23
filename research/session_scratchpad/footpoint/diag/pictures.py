"""Picture check (plan.json, D): per window, one frame, the truth box (green)
and ours (red) on six players spread across the panorama, enlarged; dots on
the two foot points.

    python pictures.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
import eval_soccertrack_v2 as ev  # noqa: E402

WIN = ROOT / "data/soccertrack_v2"
GPU = ROOT / ".cache/gpu"
OUT = HERE / "pictures"
WINDOWS = ["st2_117093_1st_f015000", "st2_118575_1st_f015000", "st2_118576_1st_f030000", "st2_118578_1st_f030000",
           "st2_118578_2nd_f015000", "st2_128058_1st_f030000", "st2_132877_1st_f015000"]
FRAME = 1500
HALF, ZOOM = 36, 5


def frame_at(path: Path, n: int) -> np.ndarray:
    """Decoded in order (no seek, so no seek error)."""
    cap = cv2.VideoCapture(str(path))
    for _ in range(n):
        cap.grab()
    ok, img = cap.read()
    cap.release()
    assert ok, path
    return img


def main():
    OUT.mkdir(exist_ok=True)
    notes = {}
    for w in WINDOWS:
        info = json.loads((WIN / f"{w}.json").read_text())
        m = pd.read_parquet(GPU / w / "metric_v3s_link.parquet")
        p = m[(m.cls == "player") & (m.frame == FRAME)].copy()
        truth = ev.truth(WIN / f"{w}_gt.parquet")
        truth = truth[truth.frame == FRAME]
        b = ev.boxes(p)
        b["person"] = ev.match_feet(b, truth)
        b = b.dropna(subset=["person"]).sort_values("px_raw")
        pick = b.iloc[np.unique(np.linspace(0, len(b) - 1, 6).round().astype(int))]
        img = frame_at(Path(info["path"].replace("/home/user/Veo", str(ROOT))), FRAME)
        tiles, rec = [], []
        for _, r in pick.iterrows():
            t = truth[truth.person == r.person].iloc[0]
            cu, cv = int((t.x0 + t.x1) / 2), int(t.y1 - 18)
            x0, y0 = max(cu - HALF, 0), max(cv - HALF, 0)
            crop = cv2.resize(img[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF], None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)

            def z(x, y):
                return int((x - x0) * ZOOM), int((y - y0) * ZOOM)
            cv2.rectangle(crop, z(t.x0, t.y0), z(t.x1, t.y1), (0, 255, 0), 2)
            cv2.rectangle(crop, z(r.x0, r.y0), z(r.x1, r.y1), (0, 0, 255), 2)
            cv2.circle(crop, z((t.x0 + t.x1) / 2, t.y1), 6, (0, 255, 0), -1)
            cv2.circle(crop, z(r.px_raw, r.py_raw), 6, (0, 0, 255), -1)
            cv2.putText(crop, f"u={int(r.px_raw)}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            tiles.append(cv2.copyMakeBorder(crop, 0, 2 * HALF * ZOOM - crop.shape[0], 0, 2 * HALF * ZOOM - crop.shape[1],
                                            cv2.BORDER_CONSTANT, value=0))
            rec.append({"u": round(float(r.px_raw), 1), "du": round(float((t.x0 + t.x1) / 2 - r.px_raw), 1),
                        "dv": round(float(t.y1 - r.py_raw), 1), "truth h": round(float(t.y1 - t.y0), 1), "our h": round(float(r.crop_h), 1)})
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        cv2.imwrite(str(OUT / f"{w}.jpg"), np.vstack(rows))
        notes[w] = {"match": info["paper_label"], "frame": FRAME, "players": rec}
        print(w, info["paper_label"], rec, flush=True)
    (OUT / "pictures.json").write_text(json.dumps(notes, indent=1))


if __name__ == "__main__":
    main()
