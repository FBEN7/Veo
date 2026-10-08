"""One name per player across every clip of a match.

`player_identity` makes people within a clip, where two tracks on screen
together are two people. Across clips there is no such evidence -- clips
are minutes apart -- so the link is appearance alone: each clip's
identities carry their mean look (`identify_players.py`), and identities
of different clips are joined most-alike first.

Two things are fixed first:

- **teams**: "team_A" in one clip is not "team_A" in the next -- team
  assignment names its clusters per clip. Each clip's two teams are
  matched to the first clip's by the mean look of their players, keeping
  or swapping the labels, whichever is more alike;
- **who counts**: an identity seen on fewer than `MIN_FRAMES` frames has a
  look too uncertain to join on, and stays its clip's own.

Then average-link clustering within each team: two clusters merge while
their mean pairwise look is at least `threshold`, and never when they hold
two identities of one clip that are on screen together -- the rule within
a clip. Forbidding any two identities of one clip, as first written, kept
a person's pieces in one clip from ever joining and left 13% of the true
links on held-out clips.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MIN_FRAMES = 50
MAX_SHARED_FRAMES = 2
# An unplaced identity this alike to a team's mean look joins that team.
OTHER_TO_TEAM = 0.5
# Teams from the looks across the whole match (two clusters) instead of
# each clip's team labels lined up.
# On ten game-state clips of one game, threshold 0.8: lining up each
# clip's team labels gave precision 75% / recall 48% on the five clips
# used to choose and 56% / 17% on the other five; teams by look, 69% / 44%
# and 69% / 34%. Chosen for holding up on clips it was not chosen on.
TEAMS_BY_LOOK = True
# Shirt numbers read by `read_numbers_parseq.py` (`numbers.json`), where
# present: identities of a team with the same number are one player, two
# with different numbers are two.
USE_NUMBERS = True
TEAM_RESTARTS = 20
THRESHOLD = 0.8
TEAMS = ("team_A", "team_B")


def load(out_dirs) -> list[dict]:
    """Every identity of every clip with a look and enough frames."""
    rows = []
    for d in map(Path, out_dirs):
        from .track_split import player_tracks

        blob = json.loads((d / "identities.json").read_text())
        nf = d / "numbers.json"
        numbers = json.loads(nf.read_text()) if nf.exists() else {}
        tracks = player_tracks(d)
        tracks = tracks[tracks.cls == "player"]
        seen = {int(t): set(g.frame.astype(int))
                for t, g in tracks.groupby("track_id")}
        for i in blob["identities"]:
            if i.get("look") is None:
                continue
            on = set().union(*(seen.get(int(t), set()) for t in i["tracks"]))
            num = numbers.get(str(i["key"]), {}) if USE_NUMBERS else {}
            rows.append({"clip": d.name, "key": i["key"], "team": i["team"],
                         "frames": i["frames"], "on": on,
                         "number": num.get("number"),
                         "share": num.get("share", 0.0),
                         "look": np.array(i["look"], dtype=float)})
    return rows


def align_teams(rows) -> dict:
    """clip -> {its team label: the reference clip's label}."""
    clips = sorted({r["clip"] for r in rows})
    def team_look(clip, team):
        vs = [r["look"] * r["frames"] for r in rows
              if r["clip"] == clip and r["team"] == team]
        if not vs:
            return None
        v = np.sum(vs, axis=0)
        return v / max(np.linalg.norm(v), 1e-9)
    ref = {t: team_look(clips[0], t) for t in TEAMS}
    out = {}
    for clip in clips:
        mine = {t: team_look(clip, t) for t in TEAMS}
        if any(v is None for v in list(mine.values()) + list(ref.values())):
            out[clip] = {t: t for t in TEAMS}
            continue
        keep = mine["team_A"] @ ref["team_A"] + mine["team_B"] @ ref["team_B"]
        swap = mine["team_A"] @ ref["team_B"] + mine["team_B"] @ ref["team_A"]
        out[clip] = ({t: t for t in TEAMS} if keep >= swap
                     else {"team_A": "team_B", "team_B": "team_A"})
    return out


def _two_teams(rows, iters: int = 20, restarts: int = TEAM_RESTARTS) -> dict:
    """Split identities into two teams by look, over every clip at once:
    frame-weighted 2-means on the unit looks, started `restarts` times
    from an identity seen on at least `MIN_FRAMES` frames and the one least
    like it, keeping the tightest split.

    Seeded once with the two least alike of all identities, as first
    written, the seeds at Stoke v Huddersfield were two 11- and 13-frame
    identities nobody had placed, and the split was those few against
    everyone: 27 of 47 players a person named were on the right side;
    with restarts, 39 (Reading v Fulham: 50 of 55 either way)."""
    looks = np.stack([r["look"] for r in rows])
    w = np.array([r["frames"] for r in rows], dtype=float)
    long_ = np.where(w >= MIN_FRAMES)[0]
    if not len(long_):
        long_ = np.arange(len(rows))
    rng = np.random.default_rng(0)
    best = None
    for k in range(restarts):
        a = long_[np.argmax(w[long_])] if k == 0 else long_[rng.integers(len(long_))]
        b = long_[np.argmin(looks[long_] @ looks[a])]
        c = np.stack([looks[a], looks[b]])
        for _ in range(iters):
            lab = np.argmax(looks @ c.T, axis=1)
            for j in range(2):
                if (lab == j).any():
                    v = (looks[lab == j] * w[lab == j, None]).sum(axis=0)
                    c[j] = v / max(np.linalg.norm(v), 1e-9)
        lab = np.argmax(looks @ c.T, axis=1)
        fit = float((w * (looks @ c.T).max(axis=1)).sum())
        if best is None or fit > best[0]:
            best = (fit, lab)
    return {id(r): TEAMS[int(best[1][i])] for i, r in enumerate(rows)}


def link(rows, threshold: float = THRESHOLD,
         min_frames: int = MIN_FRAMES, per_team: int | None = None) -> dict:
    """(clip, identity key) -> match-wide player id."""
    if TEAMS_BY_LOOK:
        teams = {r["clip"]: {} for r in rows}
        group = _two_teams(rows)
    else:
        teams = align_teams(rows)
        group = None
    # Identities the team assignment left unplaced ("other") -- on held-out
    # game-state clips up to 20 a clip -- join the team they look most like,
    # if they clearly look like one; the rest (referees, among others) are
    # linked among themselves.
    if group is None:
        group = {id(r): teams[r["clip"]].get(r["team"]) for r in rows}
    centre = {}
    for team in TEAMS:
        vs = [r["look"] * r["frames"] for r in rows if group[id(r)] == team]
        if vs:
            v = np.sum(vs, axis=0)
            centre[team] = v / max(np.linalg.norm(v), 1e-9)
    for r in rows:
        if group[id(r)] is None and centre:
            sims = {t: float(r["look"] @ c) for t, c in centre.items()}
            t = max(sims, key=sims.get)
            group[id(r)] = t if sims[t] >= OTHER_TO_TEAM else "other"
    out, next_id = {}, 0
    for team in TEAMS + ("other",):
        members = [r for r in rows if group[id(r)] == team
                   and (r["frames"] >= min_frames
                        or r.get("number") is not None)]
        n = len(members)
        if n:
            looks = np.stack([m["look"] for m in members])
            sim = looks @ looks.T
        clash = np.zeros((n, n), dtype=bool)
        for a in range(n):
            for b in range(a + 1, n):
                if (members[a]["clip"] == members[b]["clip"]
                        and len(members[a]["on"] & members[b]["on"])
                        > MAX_SHARED_FRAMES):
                    clash[a, b] = clash[b, a] = True
        clusters = _by_number(members, clash)
        num = [_number(members, c) for c in clusters]
        while True:
            if per_team is not None and team in TEAMS:
                # Stop at a team's number of players instead of a likeness.
                if len(clusters) <= per_team:
                    break
                best, pair = -np.inf, None
            else:
                best, pair = threshold, None
            for x in range(len(clusters)):
                for y in range(x + 1, len(clusters)):
                    if clash[np.ix_(clusters[x], clusters[y])].any():
                        continue
                    if (num[x] is not None and num[y] is not None
                            and num[x] != num[y]):
                        continue    # two numbers read: two players
                    s = sim[np.ix_(clusters[x], clusters[y])].mean()
                    if s > best:
                        best, pair = s, (x, y)
            if pair is None:
                break
            x, y = pair
            clusters[x] += clusters[y]
            clusters.pop(y)
            num[x] = num[x] if num[x] is not None else num[y]
            num.pop(y)
        for c in clusters:
            for i in c:
                out[(members[i]["clip"], members[i]["key"])] = next_id
            next_id += 1
    for r in rows:
        if (r["clip"], r["key"]) not in out:
            out[(r["clip"], r["key"])] = next_id
            next_id += 1
    return out


def _number(members, cluster):
    nums = {members[i].get("number") for i in cluster} - {None}
    return nums.pop() if len(nums) == 1 else None


def _by_number(members, clash):
    """Starting clusters: identities with the same read number together,
    surest read first, never two on screen at once (then one number is
    wrong, and the less sure stays alone); the rest one each."""
    clusters, by_num = [], {}
    order = sorted(range(len(members)),
                   key=lambda i: -(members[i].get("share") or 0.0))
    for i in order:
        n = members[i].get("number")
        home = by_num.get(n) if n is not None else None
        if home is not None and not clash[np.ix_(clusters[home], [i])].any():
            clusters[home].append(i)
            continue
        if n is not None and home is None:
            by_num[n] = len(clusters)
        clusters.append([i])
    return clusters


def write(out_dirs) -> dict:
    """Name every identity of these clips (one match) and write
    `match_identities.json` beside each clip's identities."""
    rows = load(out_dirs)
    names = link(rows)
    for d in map(Path, out_dirs):
        mine = {str(k): int(v) for (clip, k), v in names.items()
                if clip == d.name}
        (d / "match_identities.json").write_text(json.dumps(mine, indent=1))
    return names


def names_of(out_dir) -> dict:
    """track id -> the player's name: match-wide (`match_identities.json`)
    where linked across the match, else the clip's identity."""
    out_dir = Path(out_dir)
    idf = out_dir / "identities.json"
    if not idf.exists():
        return {}
    ident = {int(k): v["identity"]
             for k, v in json.loads(idf.read_text())["tracks"].items()}
    mf = out_dir / "match_identities.json"
    if mf.exists():
        match = json.loads(mf.read_text())
        return {t: f"player M{match[str(i)]}" if str(i) in match
                else f"player {i}" for t, i in ident.items()}
    return {t: f"player {i}" for t, i in ident.items()}
