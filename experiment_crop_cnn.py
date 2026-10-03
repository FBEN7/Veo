"""Does a crop CNN transfer where a colour classifier did not?

The Roboflow route rests on an assumption: that a model which sees the crop --
shape, posture, texture, kit against skin against grass -- generalises across
matches where one that sees only a colour histogram does not. The histogram
version is already measured and fails, fitting its own match at AUC 1.00 and
transferring at 0.65 and 0.87 against the distance rule's 0.91 and 0.97.

That assumption can be tested before any dataset is fetched, because the
hand-read kit labels already say which tracks are players. It is a small test
-- 140 player tracks and 46 non-player tracks across two matches -- and the
architecture is the same `SmallNet` that `train_player_filter.py` would train
on Roboflow crops. Train on one match, test on the other.

What it can show: that the approach transfers, which is the case for spending
effort on the dataset. What it cannot show: how well, since two matches of
touchline staff is not a sample of the world's touchline staff. A negative
here would be the more informative result -- it would mean a bigger dataset
of the same kind of input is not obviously the answer.

Provenance: trains on SoccerNet-derived crops, so the model cannot ship. It
is an instrument for deciding what to build.

    python experiment_crop_cnn.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyse_pass_outcome import WINDOWS
from sweep_roster_rejection import LABELLED, load as load_window
from train_player_filter import SmallNet, auc, crops_for_tracks, to_tensor

# More crops per track than the pipeline needs, because the track count is the
# binding constraint here and each track can supply many views.
PER_TRACK = 16
EPOCHS = 30


def build(window):
    """Crops per labelled track, with player/non-player truth."""
    name, df, _, kit, nonplayers = load_window(window)
    out_dir = next(w[1] for w in WINDOWS if w[0] == window)
    clip = json.loads((Path(out_dir) / "clip.json").read_text())["path"]
    raw = pd.read_parquet(Path(out_dir) / "tracks_grass.parquet")
    by_track = crops_for_tracks(clip, raw, set(kit) | set(nonplayers),
                                per_track=PER_TRACK)

    tids = [t for t in by_track if t in kit or t in nonplayers]
    resid = np.array([float(df.loc[t, "residual"]) for t in tids])
    is_player = np.array([t in kit for t in tids])
    return name, tids, by_track, is_player, resid


def train_on(by_track, tids, is_player, seed=0):
    """Fit SmallNet at crop level; positive class is 'is a player'."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X = np.concatenate([by_track[t] for t in tids])
    y = np.concatenate([np.full(len(by_track[t]), p, dtype=np.float32)
                        for t, p in zip(tids, is_player)])
    Xt, yt = to_tensor(X), torch.from_numpy(y)

    model = SmallNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    pos = float(yt.sum())
    lossfn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(len(yt) - pos) / max(pos, 1.0)]))

    n, batch = len(Xt), 128
    for _ in range(EPOCHS):
        model.train()
        order = torch.randperm(n)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xb = Xt[idx]
            if rng.random() < 0.5:
                xb = torch.flip(xb, dims=[3])
            opt.zero_grad()
            lossfn(model(xb), yt[idx]).backward()
            opt.step()
    model.eval()
    return model


def score(model, by_track, tids):
    """Median crop score per track."""
    out = []
    with torch.no_grad():
        for t in tids:
            s = torch.sigmoid(model(to_tensor(by_track[t]))).numpy()
            out.append(float(np.median(s)))
    return np.array(out)


def main():
    data = {w: build(w) for w in LABELLED}
    for w in LABELLED:
        name, tids, _, is_player, _ = data[w]
        print(f"{name}: {len(tids)} labelled tracks "
              f"({int(is_player.sum())} players, "
              f"{int((~is_player).sum())} not)")

    print("\nPositive class is 'IS a player'. Higher AUC is better.\n")
    print(f"  {'trained on':18s} {'tested on':18s} {'crop CNN':>9s} "
          f"{'colour hist':>12s} {'distance rule':>14s}")

    # Colour-histogram numbers from experiment_player_classifier.py, which
    # scores the opposite polarity (positive = non-player); the AUCs are
    # symmetric, so they are quoted here as measured.
    hist = {"w1 stoke 1820": 0.65, "reading 5115": 0.87}

    for train_w in LABELLED:
        test_w = [w for w in LABELLED if w != train_w][0]
        ntr, ttr, btr, ptr, _ = data[train_w]
        nte, tte, bte, pte, rte = data[test_w]

        model = train_on(btr, ttr, ptr)
        s = score(model, bte, tte)
        print(f"  {ntr:18s} {nte:18s} {auc(s, pte):9.2f} "
              f"{hist.get(nte, float('nan')):12.2f} "
              f"{auc(-rte, pte):14.2f}")

    print("\nOn its own training match (optimistic -- it has seen these):")
    for w in LABELLED:
        name, tids, by_track, is_player, _ = data[w]
        model = train_on(by_track, tids, is_player)
        print(f"  {name:18s} {auc(score(model, by_track, tids), is_player):.2f}")


if __name__ == "__main__":
    main()
