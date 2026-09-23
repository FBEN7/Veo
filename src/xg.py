"""Expected goals, fitted rather than borrowed.

The blocker on xG here was never the arithmetic. It was that an xG figure is
a set of coefficients, and coefficients have to come from somewhere: a
published model whose numbers can be cited, or a labelled dataset of shots
with their outcomes. Writing down plausible-looking numbers produces a figure
that reads like Opta and is not, which is worse than having no figure.

xGHub (github.com/mguti97/xGHub, CVIU, "Once Upon a Goal: Towards
orientation-based shot metrics in football") is the second kind. It is a
dataset, not a model -- the repository holds a README and nothing else -- and
every shot in it carries `is_goal` alongside its geometry. So the model is
fitted here, on that data, and the fit is reported with its calibration
rather than asserted.

## What we can feed it, and what we cannot

xGHub's contribution is body orientation: six of its fields describe which
way the shooter was facing in 2-D and 3-D. This pipeline has no pose
estimation, so none of them is available, and neither is `body_part` or
`technique`. What is available is the classical geometry -- distance and the
angle the goal mouth subtends -- which `shot_geometry.py` computes exactly,
plus a rough analogue of defensive occlusion.

So what is fitted here is a **reduced model**: the classical part of a paper
whose point was the part we cannot compute. That is a legitimate use of the
data and a poor summary of the paper, and any figure this produces should be
described as the former.

## The distance convention is resolved, not assumed

xGHub documents `distance_from_goal` as "Euclidean distance to the goal
line", which is ambiguous: it could be the perpendicular distance to the line
or the distance to the middle of the goal. Those differ by a factor that
grows with how wide the shot is, and picking wrong would bias every xG
systematically while looking entirely reasonable -- the class of error this
project has been caught by more than once.

It does not have to be guessed. The dataset records three geometric
quantities for what is only two degrees of freedom, so the third
over-determines the pair: reconstruct the shot position from a distance and a
bearing under each convention, recompute the angle the goal mouth subtends,
and compare with the recorded one. The wrong convention does not reproduce
it. `infer_distance_convention` does this and reports the residual for both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import shot_geometry as sg

# The two readings of "distance from goal" that the documentation allows.
CONVENTIONS = ("goal_centre", "perpendicular")

# Residual above which a convention is rejected, in degrees of reconstructed
# goal-mouth angle. The right convention reproduces the recorded angle to
# within rounding; the wrong one is out by tens of degrees on wide shots.
CONVENTION_MAX_RESIDUAL_DEG = 1.0

# The features fitted by default: the two the geometry gives exactly.
DEFAULT_FEATURES = ("distance_m", "angle_rad")

# Shots are held out by match, not at random. Two shots from the same game
# share a pitch, a camera, a pair of teams and often a passage of play, so
# splitting within a match lets the model see its own test conditions.
HELD_OUT_FRACTION = 0.25


@dataclass
class XGModel:
    """Fitted coefficients, and the evidence for them."""

    features: tuple
    coefficients: list
    intercept: float
    n_train: int
    n_test: int
    base_rate: float
    metrics: dict = field(default_factory=dict)
    source: str = ""
    convention: str = ""
    note: str = ("Reduced model: classical shot geometry only. The "
                 "orientation features that are xGHub's contribution are not "
                 "available from this pipeline.")

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "XGModel":
        data = json.loads(Path(path).read_text())
        data["features"] = tuple(data["features"])
        return cls(**data)

    def predict(self, **features) -> np.ndarray:
        """Probability a shot with these features becomes a goal."""
        missing = [f for f in self.features if f not in features]
        if missing:
            raise ValueError(f"missing features: {missing}")
        x = np.column_stack([np.asarray(features[f], dtype=float)
                             for f in self.features])
        z = self.intercept + x @ np.asarray(self.coefficients, dtype=float)
        return 1.0 / (1.0 + np.exp(-z))

    def summary(self) -> str:
        terms = "  ".join(f"{f} {c:+.4f}"
                          for f, c in zip(self.features, self.coefficients))
        m = self.metrics
        return (f"  features     {terms}  intercept {self.intercept:+.4f}\n"
                f"  fitted on    {self.n_train} shots, held out {self.n_test}\n"
                f"  base rate    {self.base_rate:.3f}\n"
                f"  held out     Brier {m.get('brier', float('nan')):.4f} "
                f"(base {m.get('brier_base', float('nan')):.4f}), "
                f"log loss {m.get('log_loss', float('nan')):.4f}, "
                f"AUC {m.get('auc', float('nan')):.3f}")


# --- reading xGHub -----------------------------------------------------------

def load_xghub(root: Path) -> pd.DataFrame:
    """Every shot's tabular metadata, as one frame.

    Expects the published layout: `labels/<id>/tabular_data.json`.
    """
    root = Path(root)
    labels = root / "labels" if (root / "labels").is_dir() else root
    rows = []
    for meta in sorted(labels.glob("*/tabular_data.json")):
        try:
            data = json.loads(meta.read_text())
        except json.JSONDecodeError:
            continue
        data["shot_id"] = meta.parent.name
        rows.append(data)
    if not rows:
        raise SystemExit(
            f"no tabular_data.json files under {labels}. Expected the "
            "published layout, labels/<id>/tabular_data.json.")
    return pd.DataFrame(rows)


def geometry_from(distance, bearing_deg, convention: str):
    """Recover (forward, lateral) from a distance and a bearing.

    `forward` is metres out from the goal line, `lateral` is offset from the
    middle of the goal -- the frame `shot_geometry` works in.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}")
    distance = np.asarray(distance, dtype=float)
    bearing = np.radians(np.asarray(bearing_deg, dtype=float))
    if convention == "goal_centre":
        return distance * np.cos(bearing), distance * np.sin(bearing)
    # perpendicular: the recorded distance is the forward leg itself
    return distance, distance * np.tan(bearing)


def _mouth_angle_from(forward, lateral):
    """`shot_geometry.goal_mouth_angle`, in the goal frame directly."""
    numerator = sg.GOAL_WIDTH_M * forward
    denominator = forward ** 2 + lateral ** 2 - sg.GOAL_HALF_M ** 2
    return np.where(forward >= 0.0, np.arctan2(numerator, denominator), 0.0)


def infer_distance_convention(df: pd.DataFrame,
                              distance_col: str = "distance_from_goal",
                              bearing_col: str = "angle_to_goal",
                              mouth_col: str = "angle_to_posts",
                              verbose: bool = True):
    """Which reading of the distance column reproduces the recorded angle.

    Returns (convention, residuals). The dataset gives three geometric
    numbers for two degrees of freedom, so the third is a check rather than
    an input: reconstruct the position from distance and bearing, recompute
    how wide the goal looks, and see which convention agrees.
    """
    for col in (distance_col, bearing_col, mouth_col):
        if col not in df.columns:
            raise ValueError(f"column {col!r} not in the dataset")

    recorded = np.abs(np.asarray(df[mouth_col], dtype=float))
    residuals = {}
    for convention in CONVENTIONS:
        forward, lateral = geometry_from(df[distance_col], df[bearing_col],
                                         convention)
        rebuilt = np.degrees(_mouth_angle_from(forward, lateral))
        diff = np.abs(rebuilt - recorded)
        residuals[convention] = float(np.nanmedian(diff))

    best = min(residuals, key=residuals.get)
    if verbose:
        for convention, value in residuals.items():
            mark = " <-" if convention == best else ""
            print(f"    {convention:<14s} median residual "
                  f"{value:7.3f} deg{mark}")
    if residuals[best] > CONVENTION_MAX_RESIDUAL_DEG:
        raise SystemExit(
            f"neither reading of {distance_col!r} reproduces {mouth_col!r} "
            f"(best {best} at {residuals[best]:.2f} deg). The geometry is not "
            "what the documentation describes; do not fit until this is "
            "understood, or every xG will carry the error.")
    return best, residuals


def features_from_xghub(df: pd.DataFrame, convention: str) -> pd.DataFrame:
    """xGHub rows as the features this pipeline can actually compute."""
    forward, lateral = geometry_from(df["distance_from_goal"],
                                     df["angle_to_goal"], convention)
    out = pd.DataFrame({
        "distance_m": np.hypot(forward, lateral),
        "angle_rad": _mouth_angle_from(forward, lateral),
        "forward_m": forward,
        "lateral_m": lateral,
        "is_goal": df["is_goal"].astype(int),
    })
    for extra in ("match_code", "shot_id", "goal_face_occlusion_joint"):
        if extra in df.columns:
            out[extra] = df[extra].to_numpy()
    return out


# --- fitting -----------------------------------------------------------------

def _split_by_match(df: pd.DataFrame, seed: int = 0):
    """Hold out whole matches, so no shot's own game trains the model."""
    if "match_code" not in df.columns:
        rng = np.random.default_rng(seed)
        mask = rng.random(len(df)) >= HELD_OUT_FRACTION
        return df[mask], df[~mask]

    matches = np.array(sorted(df.match_code.dropna().unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(matches)
    n_held = max(1, int(round(len(matches) * HELD_OUT_FRACTION)))
    held = set(matches[:n_held])
    is_held = df.match_code.isin(held)
    return df[~is_held], df[is_held]


def calibration_table(y_true, y_prob, bins: int = 5) -> pd.DataFrame:
    """Predicted against observed, which is the test that matters for xG.

    A model can rank shots well and still be wrong about their level, and xG
    is consumed as a level -- "0.6 expected goals" is a quantity, not an
    ordering. So the reliability table is reported alongside AUC rather than
    instead of it.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.quantile(y_prob, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (y_prob > lo) & (y_prob <= hi)
        if not sel.any():
            continue
        rows.append(dict(n=int(sel.sum()),
                         predicted=float(y_prob[sel].mean()),
                         observed=float(y_true[sel].mean())))
    return pd.DataFrame(rows)


def fit(df: pd.DataFrame, features=DEFAULT_FEATURES, seed: int = 0,
        source: str = "", convention: str = "",
        verbose: bool = True) -> XGModel:
    """Fit a logistic model and report how well it holds up out of sample."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

    features = tuple(features)
    usable = df.dropna(subset=list(features) + ["is_goal"])
    train, test = _split_by_match(usable, seed)
    if len(train) < 50 or len(test) < 20:
        raise SystemExit(
            f"too few shots to fit and check: {len(train)} train, "
            f"{len(test)} held out. A two-feature logistic needs hundreds "
            "before its coefficients mean anything.")

    # Unregularised on purpose. scikit-learn applies L2 by default, which is
    # the right default when a model is only going to be used for prediction
    # and the wrong one here: these coefficients get written down, reported
    # and compared against published models, so they should be the maximum
    # likelihood estimates rather than shrunk toward zero by a penalty
    # strength nobody chose. Measured on synthetic shots drawn from known
    # coefficients, the default penalty moved the fitted intercept 23% off
    # its true value while the slopes stayed within 6%.
    # `C=np.inf` rather than `penalty=None`: same unregularised fit, and the
    # spelling that survives scikit-learn 1.10, where `penalty` is removed.
    model = LogisticRegression(max_iter=1000, C=np.inf)
    model.fit(train[list(features)].to_numpy(float),
              train.is_goal.to_numpy(int))

    y_true = test.is_goal.to_numpy(int)
    y_prob = model.predict_proba(test[list(features)].to_numpy(float))[:, 1]
    base_rate = float(train.is_goal.mean())

    metrics = dict(
        brier=float(brier_score_loss(y_true, y_prob)),
        # The honest floor: predicting the training base rate for every shot.
        # A model that cannot beat this has learned nothing about geometry.
        brier_base=float(np.mean((y_true - base_rate) ** 2)),
        log_loss=float(log_loss(y_true, y_prob, labels=[0, 1])),
        auc=(float(roc_auc_score(y_true, y_prob))
             if len(np.unique(y_true)) > 1 else float("nan")),
    )

    fitted = XGModel(features=features,
                     coefficients=[float(c) for c in model.coef_[0]],
                     intercept=float(model.intercept_[0]),
                     n_train=int(len(train)), n_test=int(len(test)),
                     base_rate=base_rate, metrics=metrics,
                     source=source, convention=convention)
    if verbose:
        print(fitted.summary())
        print("\n  calibration on held-out shots:")
        table = calibration_table(y_true, y_prob)
        for _, row in table.iterrows():
            print(f"    n={int(row.n):4d}  predicted {row.predicted:.3f}  "
                  f"observed {row.observed:.3f}")
    return fitted
