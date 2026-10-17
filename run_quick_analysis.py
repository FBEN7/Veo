#!/usr/bin/env python3
"""Quick analysis with stride=2 for faster processing."""

import subprocess
import sys

# Run with stride=2 for 2x speed improvement
result = subprocess.run([
    sys.executable, "main_professional.py",
    "data/match_first2min.mp4",
    "--label", "2-Minute Test Video (Optimized)",
    "--target-ball-detection", "0.70",
    "--stride", "2",  # Process every 2nd frame for 2x speed
    "--debug"
], timeout=600)

sys.exit(result.returncode)
