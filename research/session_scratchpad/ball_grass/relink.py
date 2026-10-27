"""Relink every shots clip's ball without the grass test (plan.json):
output_<w>/metric_v3s_link_nograss.parquet beside the existing pitch tracks.

    python relink.py
"""
import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/session_scratchpad/ball/events_tune"))
import score_soccernet as sc  # noqa: E402
import tune_rules as tr  # noqa: E402
from src import pixel_scale  # noqa: E402
from src import soccertrack_v2 as st  # noqa: E402

SKIP = {"117092"}
NEW_PLAYERS = {"128057", "132831"}


def main():
    done = 0
    for p in sorted((ROOT / ".cache/gpu").iterdir()):
        w = p.name
        m = w.split("_")[1]
        if m in SKIP:
            continue
        base = ROOT / (".cache/gpu_nograss" if m in NEW_PLAYERS else ".cache/gpu") / w / f"output_{w}"
        run = base / "shots_v3s_link"
        if not (run / "tracks_grass.parquet").exists():
            continue
        cal = st.calibration(m)
        with contextlib.redirect_stdout(io.StringIO()):
            merged = sc.merged_from_cached(run, cal, select="link", top_k=4, tau=0.10,
                                           link_params=tr.LINK, ball_grass_filter=False)
        metric = pixel_scale.to_pitch_metres(merged, lambda uv: st.image_to_pitch(uv, cal))
        metric.to_parquet(base / "metric_v3s_link_nograss.parquet")
        done += 1
        print(w, int((metric.cls == "ball").sum()), flush=True)
    print("relinked", done)


if __name__ == "__main__":
    main()
