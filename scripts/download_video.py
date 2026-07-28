"""Download a public amateur football video for pipeline testing.

Usage:
    python scripts/download_video.py
    python scripts/download_video.py --url <youtube_url> --out data/match.mp4
    python scripts/download_video.py --list   # show preset options

The default target is a freely available amateur football match video.
Resolution is capped at 720p to keep file size manageable (~500 MB for 90 min).
"""
import argparse
import subprocess
import sys
from pathlib import Path

# Curated list of public-domain or Creative Commons amateur football videos.
# Each entry has a URL and approximate duration.
PRESETS = [
    {
        "name": "Amateur 11v11 full match (YouTube, ~75 min, 720p)",
        "url": "https://www.youtube.com/watch?v=VqnZB4UvFCA",
        "note": "Fixed-camera amateur match, good contrast between kits.",
    },
    {
        "name": "Amateur 11v11 full match — alternative (YouTube, ~90 min, 720p)",
        "url": "https://www.youtube.com/watch?v=wUZj0DX26uQ",
        "note": "Side-angle tripod camera, green pitch, clearly visible ball.",
    },
]


def _check_ytdlp():
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        print("yt-dlp not found. Installing …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp>=2024.1.0"])


def download(url: str, out_path: str = "data/match.mp4"):
    _check_ytdlp()
    import yt_dlp  # noqa: E402

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        # Best video+audio merged, capped at 720p, prefer mp4
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "merge_output_format": "mp4",
        "outtmpl": str(out.with_suffix("")) + ".%(ext)s",
        "quiet": False,
        "no_warnings": False,
        "progress": True,
    }

    print(f"\nDownloading: {url}")
    print(f"Destination: {out}\n")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        duration = info.get("duration", 0)
        title = info.get("title", "unknown")

    print(f"\n✓ Done — '{title}' ({duration // 60} min {duration % 60} s)")
    print(f"  File: {out}")
    print(
        "\nNext steps:\n"
        "  1. python -m src.calibrate data/match.mp4   # click 4 pitch corners\n"
        "  2. python main.py data/match.mp4 --label 'My Team vs Opponent'\n"
    )


def list_presets():
    print("\nAvailable preset videos:\n")
    for i, p in enumerate(PRESETS, 1):
        print(f"  [{i}] {p['name']}")
        print(f"      URL  : {p['url']}")
        print(f"      Note : {p['note']}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Download a public football video for the analysis pipeline."
    )
    ap.add_argument(
        "--url",
        default=PRESETS[0]["url"],
        help="YouTube URL to download (default: first preset).",
    )
    ap.add_argument(
        "--out",
        default="data/match.mp4",
        help="Output file path (default: data/match.mp4).",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List available preset URLs and exit.",
    )
    ap.add_argument(
        "--preset",
        type=int,
        choices=range(1, len(PRESETS) + 1),
        metavar=f"1-{len(PRESETS)}",
        help="Download a specific preset by number instead of providing a URL.",
    )
    args = ap.parse_args()

    if args.list:
        list_presets()
        return

    url = PRESETS[args.preset - 1]["url"] if args.preset else args.url
    download(url=url, out_path=args.out)


if __name__ == "__main__":
    main()
