# Team assignment: the number was measuring something else

Team assignment was the binding constraint on pass outcome. The receiver
rebuild in `src/events.py` reaches balanced accuracy 0.83 on a held-out match
when the passer's team is taken from SoccerNet, and 0.28 when it comes from
our kit clustering. So the clustering was the thing to fix, and it measured
0.55 to 0.81 across four windows.

**Kit clustering measures 0.93.** It was never the problem.

## The metric was compound

Every team-assignment figure in this project was scored at matched labelled
passes: our team for the passer against SoccerNet's side for that action.
That measures

    (did we cluster this track into the right team)
      x (did we attribute the event to the right player)

and it cannot separate the two. The second factor is broken badly enough to
dominate.

The tell was there and went unread for a long time. Nine colour
representations -- linear HSV, circular hue, brightness removed, brightness
normalised by the frame, grass removed, a hue-saturation histogram -- and six
values of k all scored between 0.64 and 0.75. When every variant of a
component returns the same number, the number is not about that component.

Three attempts to explain the 0.55 on the Reading window failed first:

* **Not an unbalanced cluster.** Reading's split is the most balanced of the
  four, 51 tracks against 52, with only 9 tracks in 'other'. A clean split,
  simply not along team lines.
* **Not illumination.** Dropping brightness entirely and normalising it by
  the frame's median both score 0.71, the same as keeping it.
* **Not multi-coloured kits.** A histogram, which represents a hooped shirt
  as two colours rather than their average, scores 0.69.

## Looking at the tracks

The tracks were rendered as strips of crops and read by eye -- 107 tracks of
at least 15 frames in the Reading window, 84 in the Stoke window. Both
matches separate trivially by eye: Fulham maroon against Reading blue-and-
white hoops, Stoke red-and-white stripes against Huddersfield navy.

Two things were immediately visible that no statistic had surfaced.

**A quarter of the "players" are not players.** Of 107 Reading tracks, 25 are
stewards in hi-vis lime, staff in dark coats, people in the crowd, and one
track that is an advertising hoarding. Sixteen of them are given a team. They
carry 16% of the labelled player-frames and take **22% of the passer and
receiver slots on matched passes** -- slots that cannot be right by kit at
all. Two of them were the credited possessor at a labelled action.

**The possession-derived truth is contaminated.** Both kits appear under the
same truth side, which is what made the compound metric unreadable.

Scored against the kits directly, on the tracks that are players:

| | possession-derived | hand-read kits |
|---|---|---|
| Reading window | 0.55 | **0.93** |

38 of 40 hooped tracks and 38 of 42 maroon tracks are on the right side.

## Three cheap rejections, all refuted

Non-players are on the pitch surface, they move, and they are a plausible
size, so the obvious filters do nothing:

| signal | non-players | real players |
|---|---|---|
| grass underfoot | 0.99 | 0.93 |
| bounding-box extent | 32-1204 px | 48-1288 px |
| median height | 57 px | 72 px |

Grass is *backwards*: a cut at any threshold removes the advertising hoarding
and nothing else. Movement and size overlap so heavily that a cut removing 2
of 8 non-players costs a real player.

## What was adopted, and what was not

Against the hand-read labels the colour representation does matter, which the
compound metric had hidden:

| feature | Reading | Stoke |
|---|---|---|
| linear HSV mean (was shipped) | 0.93 | 0.97 |
| circular hue | 0.98 | 0.98 |
| circular hue, grass pixels discarded | **0.99** | **0.98** |

`src/team_assignment_v2.py` is now the shipped implementation, with green
pixels discarded before the shirt colour is measured. That file previously
recorded "it is no more accurate" -- 0.775 against 0.778. **That conclusion
was wrong**, and wrong because of the compound metric, not because the
measurement was sloppy.

**Raising k was measured and rejected.** The shipped rule is k=3 with the
smallest cluster discarded, which allows exactly one non-team group where
there are several. Taking the two largest of more clusters lets any number
fall out, and on the window it was chosen on it works well -- purity 0.84 to
0.99 at unchanged coverage. It does not replicate:

| | purity (fitted) | purity (held out) |
|---|---|---|
| shipped k=3 | 0.84 | 0.83 |
| k=5, two largest | 0.99 | 0.85 |

and run end to end it halved team attribution on a third window, 0.80 to
0.55, splitting one side 103 tracks against 38. A rule that assumes the two
teams are the two biggest colour groups fails whenever a team fragments into
several, and nothing detects that. k stays at 3, where "two largest" and "all
but the smallest" are the same rule, so only the feature changed.

## What this bought, honestly

On the only uncontaminated measurement, kit accuracy rises 0.93 to 0.99 and
0.97 to 0.98. End to end it is a wash: mean outcome accuracy 56% before and
after, pass detection F1 at +/-2 s moving -0.02, +0.02, +0.02, 0.00 across
the four windows, still above chance on 4/4 at +/-1 s and 3/4 at +/-2 s.

That is the expected result. Team assignment was not the binding constraint;
the measurement said it was because the measurement was compound.

## What is actually binding

**Roster contamination.** A quarter of tracks are not players, they take 22%
of event slots, and none of grass, movement, size or clustering removes them
reliably across two matches. This is unsolved and it is now the first thing
to fix -- it corrupts possession, team shape and any per-player statistic,
not only pass outcome.

The signal that would work is the one not tried: these are people whose
*appearance class* differs, and a detector that distinguishes a footballer in
kit from a steward in hi-vis is a classification problem, not a clustering
one. Roboflow's football datasets are CC BY 4.0 and carry player, referee and
goalkeeper classes.

## Reproducing

    python score_team_assignment.py        # the decomposition, Reading window
    python sweep_nonplayer_rejection.py    # feature and k sweep on kit labels
    python sweep_team_variants.py          # the compound metric, for contrast
    python experiment_teams.py             # feature variants and their ceiling

The hand-read kit labels live in `output_soccernet*/kit_labels.json`, which is
gitignored: they are derived from SoccerNet video and stay out of the
repository like everything else that is.
