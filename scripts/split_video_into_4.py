from __future__ import annotations

from pathlib import Path
import cv2


def main() -> None:
    src = Path("output/tracking_overlay.mp4")
    if not src.exists():
        raise FileNotFoundError(src)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        raise RuntimeError("Video has no frames")

    base = frame_count // 4
    rem = frame_count % 4
    sizes = [base + (1 if i < rem else 0) for i in range(4)]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = []
    for i in range(4):
        out = Path(f"output/tracking_overlay_part{i+1}.mp4")
        if out.exists():
            out.unlink()
        writers.append(cv2.VideoWriter(str(out), fourcc, fps, (width, height)))

    part_idx = 0
    frames_in_part = 0
    target = sizes[part_idx]
    total_read = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            writers[part_idx].write(frame)
            frames_in_part += 1
            total_read += 1

            if total_read % 5000 == 0:
                print(f"[Split] progress: {total_read}/{frame_count} frames")

            if frames_in_part >= target and part_idx < 3:
                part_idx += 1
                frames_in_part = 0
                target = sizes[part_idx]
    finally:
        cap.release()
        for w in writers:
            w.release()

    print(f"[Split] Source frames: {frame_count}, read: {total_read}, fps: {fps:.2f}")
    for i, n in enumerate(sizes, start=1):
        print(f"[Split] part{i}: {n} frames")


if __name__ == "__main__":
    main()
