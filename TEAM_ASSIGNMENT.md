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

**Raising k was measured and rejected.** The rule was k=3 with the
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

## What the feature change alone bought

On the only uncontaminated measurement, kit accuracy rises 0.93 to 0.99 and
0.97 to 0.98. End to end, taken by itself, it is a wash: mean outcome
accuracy 56% before and after, pass detection F1 at +/-2 s moving -0.02,
+0.02, +0.02, 0.00 across the four windows, still above chance on 4/4 at
+/-1 s and 3/4 at +/-2 s.

That is the expected result, and it is the point. Team assignment was not the
binding constraint; the measurement said it was because the measurement was
compound. What *is* binding follows.

## Roster contamination: rejecting by distance instead of by counting

Counting clusters failed because it assumes the two teams are the two
*biggest* colour groups, and a team that fragments into several breaks it
silently. The fix does not count clusters at all.

Both kits form a tight group each, whatever colours they happen to be. A
steward in hi-vis is far from both; so is a figure in a dark coat; so is a
hoarding. **Distance to the nearer kit centre** is therefore kit-agnostic,
which is what a rule has to be to survive a change of match. It is expressed
as a multiple of the median distance among tracks inside the kit clusters, so
it does not depend on how far apart two particular kits sit in colour space.

Measured against two alternatives, on the hand-read labels:

| signal | AUC, Reading | AUC, Stoke |
|---|---|---|
| **distance to nearer kit centre** | **0.97** | **0.91** |
| how much a track's own colour varies | 0.48 | 0.33 |
| how isolated it is from other tracks | 0.60 | 0.47 |

The cut was chosen on one window and applied to the other, both directions:

| fitted on | cut | held-out purity |
|---|---|---|
| Reading | 2.00 | Stoke 0.73 → **0.91** |
| Stoke | 2.50 | Reading 0.77 → **0.96** |

2.5 is shipped: at 2.0 coverage falls to 0.90 on Stoke, and a dropped player
loses their events outright.

### What it bought

| | before | after |
|---|---|---|
| purity, Reading | 0.77 | **0.96** |
| purity, Stoke | 0.73 | **0.89** |
| coverage | 1.00 | 0.98 / 0.97 |
| kit accuracy | 0.93 / 0.97 | 0.99 / 0.98 |
| non-players given a team (Reading) | 16/25 | **3/25** |
| player-frames they carry (Reading) | 16% | **6%** |
| event slots landing on one (Reading) | 22% | **15%** |

End to end, pass detection improves slightly -- F1 at +/-2 s moves +0.02,
-0.01, +0.01, +0.03 across the four windows, still above chance on 4/4 at
+/-1 s -- and the outcome call separates the classes further: pooled, we call
"intercepted" on 46% of passes that kept the ball against **70%** of those
that lost it, where at the start of this work it was 56% against 56%.

### It is reduced, not solved

15% of event slots still land on someone who is not a player, and the ones
that survive are the ones standing nearest the play, so they are
over-represented in events relative to their number. Purity on Stoke is 0.89,
meaning one kept track in nine is a steward.

## The Roboflow route: prepared, and blocked on the data

The distance rule is *relative* -- it knows only that a steward looks unlike
both kits. A model trained on labelled football would know something
absolute: hi-vis is never a kit, an overcoat is never a kit. Roboflow's
football-players-detection set labels player, goalkeeper, referee and ball,
and is CC BY 4.0, so unlike SoccerNet it could ship.

**A colour classifier is not that model**, which is worth knowing before
fetching anything. Trained on one match's hand-read labels and tested on the
other's (`experiment_player_classifier.py`):

| trained on | tested on | classifier | distance rule |
|---|---|---|---|
| Reading | Stoke | 0.65 | **0.91** |
| Stoke | Reading | 0.87 | **0.97** |

It fits its own training match at AUC 1.00 and transfers worse than the
incumbent on both. Absolute colour memorises kits. Whatever is trained has to
see the crop and learn shape and texture, not a histogram -- and that is the
argument for real labelled data rather than against it.

**This container cannot reach roboflow.com.** Every subdomain is refused by
the egress proxy under organisation policy, as is huggingface.co; github.com
and pypi.org are allowed. The export is also far larger than the 30 MB upload
limit.

Neither blocks the work, because what is being trained is a filter over
*crops*, and crops are small. Two scripts split the job:

    prepare_player_crops.py   runs where Roboflow is reachable; writes a few
                              megabytes of 32x64 crops, small enough to upload
    train_player_filter.py    runs here; trains, and scores transfer onto the
                              hand-read kit labels

The negatives are mined rather than sampled, which is the part that matters.
A random background box is grass, and separating footballers from grass
teaches nothing -- the pipeline never asks about grass, it asks about a
person-shaped detection that might be a steward. So `prepare_player_crops.py`
runs the same COCO detector the pipeline deploys over the dataset's images
and keeps every `person` detection that matches no annotation. Roboflow's
annotators labelled the participants and ignored everyone else, so what is
left is exactly the contaminating population: touchline staff, substitutes,
stewards, ballboys, crowd. It also matches the deployment condition, since
the filter will see COCO person detections in use.

Both paths are tested end to end on a synthetic archive: training, weight
saving, and the transfer evaluation all run, with the distance-rule baseline
reading 0.97 and 0.91 as it should. Only the data is missing.

### The payoff is unproven, and worth saying before anyone fetches anything

The argument for the dataset is that a model seeing the crop -- shape,
posture, texture -- generalises where one seeing a colour histogram does not.
That assumption is testable now, because the hand-read labels say which
tracks are players. `experiment_crop_cnn.py` trains the same `SmallNet` on one
match's tracks and tests it on the other's:

| trained on | tested on | crop CNN | colour histogram | distance rule |
|---|---|---|---|---|
| Reading | Stoke | 0.47 | 0.65 | **0.91** |
| Stoke | Reading | 0.68 | 0.87 | **0.97** |

Both learned models fit their own match at AUC 1.00 and lose to the rule they
were meant to replace. The CNN is *worse* than the histogram, and one of its
scores is below chance.

This does not show that the Roboflow route fails. 186 tracks across two
matches is far too little to train a network that generalises, and the
dataset is roughly forty times the images across many matches. What it does
show is that **the payoff cannot be demonstrated with what is here**, and
that "learned appearance beats the distance rule" is currently an assumption
rather than a result. The distance rule is at purity 0.96 and 0.89 and costs
nothing to run.

## Reproducing

    python score_team_assignment.py        # the decomposition, both windows
    python sweep_roster_rejection.py       # the rejection rule and its cut
    python sweep_nonplayer_rejection.py    # feature and k sweep on kit labels
    python sweep_team_variants.py          # the compound metric, for contrast
    python experiment_teams.py             # feature variants and their ceiling
    python experiment_player_classifier.py # why a colour classifier is not it
    python experiment_crop_cnn.py         # nor, at this data scale, a crop CNN

The hand-read kit labels live in `output_soccernet*/kit_labels.json`, which is
gitignored: they are derived from SoccerNet video and stay out of the
repository like everything else that is.
