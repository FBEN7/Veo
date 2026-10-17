"""Give each player one name across every clip of a match.

Reads each clip's identities (`identify_players.py --split`, which stores
their looks) and links them across clips by look
(`src/match_identity.py`), writing `match_identities.json` in every clip.
The event detector and the scorers then name players match-wide.

    python match_players.py output_stoke_1302 output_stoke_4207 ...
"""

from __future__ import annotations

import argparse
from collections import Counter

from src import match_identity as mi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dirs", nargs="+", help="clips of one match")
    args = ap.parse_args()
    names = mi.write(args.out_dirs)
    sizes = Counter(Counter(names.values()).values())
    linked = sum(n for size, n in sizes.items() if size > 1)
    print(f"  {len(names)} identities -> {len(set(names.values()))} players; "
          f"{linked} players seen in more than one identity")


if __name__ == "__main__":
    main()
