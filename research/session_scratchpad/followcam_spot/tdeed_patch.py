"""follow-cam pilot (plan.json): T-DEED (commit d953b4d, GPL-3.0) packed for
Kaggle with four small patches, nothing else changed:

1. dataset/frame.py: the clip step int((1 - 0.9) * 100) is 9 in floating
   point; rounded to 10 so clips start on even frames (only every 2nd frame
   is extracted, stride 2 reads even frames only).
2. train_tdeed.py: the last epoch's weights saved every epoch
   (checkpoint_last.pt); the plan keeps the last epoch, no selection on M2.
3. train_tdeed.py: stop after training (infer.py scores the videos).
4. a no-op wandb module (no account, no network logging).

    python tdeed_patch.py SRC_REPO OUT_TAR
"""
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

WANDB = '''"""no-op stand-in for wandb (follow-cam pilot)."""
summary = {}


def login(*a, **k):
    return True


def init(*a, **k):
    return None


def log(*a, **k):
    return None
'''


def patch(path, old, new):
    s = path.read_text()
    if s.count(old) != 1:
        raise SystemExit(f"{path.name}: expected one {old!r}")
    path.write_text(s.replace(old, new))


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    rev = subprocess.run(["git", "-C", str(src), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    if rev != "d953b4d":
        raise SystemExit(f"T-DEED at {rev}, plan says d953b4d")
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "T-DEED"
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".git", "checkpoints"))
        patch(dst / "dataset/frame.py", "self._overlap = int((1-overlap) * clip_len)", "self._overlap = int(round((1-overlap) * clip_len))")
        patch(dst / "train_tdeed.py",
              "                if better:\n                    torch.save(\n                        model.state_dict(),\n"
              "                        os.path.join(os.getcwd(), args.save_dir, 'checkpoint_best.pt'))\n",
              "                if better:\n                    torch.save(\n                        model.state_dict(),\n"
              "                        os.path.join(os.getcwd(), args.save_dir, 'checkpoint_best.pt'))\n"
              "                torch.save(model.state_dict(), os.path.join(os.getcwd(), args.save_dir, 'checkpoint_last.pt'))\n")
        patch(dst / "train_tdeed.py", "    print('START INFERENCE')\n", "    sys.exit(0)  # follow-cam pilot: infer.py scores the videos\n    print('START INFERENCE')\n")
        (dst / "wandb.py").write_text(WANDB)
        with tarfile.open(out, "w:gz") as t:
            t.add(dst, arcname="T-DEED")
    print("packed", out)


if __name__ == "__main__":
    main()
