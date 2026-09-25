"""Check the xG fitting pipeline on shots whose answer we already know.

The dataset this is built for is behind an egress block, so none of it can be
run against the real thing yet. That is not a reason to leave the code
untested -- it is the reason to test it the way the rest of this project
tests things it cannot get ground truth for: manufacture data whose answer is
fixed by construction, and check the machinery recovers it.

Three things are checked, and the first is the one that would otherwise bite
silently.

  1. The distance convention is *inferred*, not assumed. Synthetic datasets
     are built under each reading of "distance from goal" and the inferrer
     has to name which. Getting this wrong biases every xG while looking
     entirely reasonable.

  2. Coefficients are recovered. Shots are drawn from a logistic with
     coefficients chosen here; the fit has to find them back.

  3. Geometry round-trips: a position turned into distance-and-bearing and
     back must land where it started.

  4. Penalties bypass the fit and take the fixed value, whatever geometry
     they arrive with -- and one recorded away from the spot is surfaced as
     a detection error rather than handed a confident number.

    python check_xg.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import shot_geometry as sg
from src import xg

# Coefficients the synthetic shots are drawn from. Chosen to be football-ish
# -- further out is worse, a wider goal mouth is better -- but their realism
# is beside the point: what is tested is that the fit finds these numbers.
TRUE_INTERCEPT = -0.50
TRUE_DISTANCE = -0.11      # per metre
TRUE_ANGLE = 1.80          # per radian of goal mouth

N_SHOTS = 6000
N_MATCHES = 60

# Coefficient recovery tolerance, as a fraction. Six thousand Bernoulli draws
# do not pin a logistic exactly.
COEFFICIENT_TOLERANCE = 0.20

# How far the fitted probabilities may sit from the true ones. This is the
# criterion that matters: an xG model is consumed as a level, so being right
# about the probability is the requirement and the coefficients are only how
# it gets there.
PROBABILITY_RMSE_TOLERANCE = 0.02


def synthetic_shots(n=N_SHOTS, seed=0) -> pd.DataFrame:
    """Shots at known positions, scored by a known model."""
    rng = np.random.default_rng(seed)
    forward = rng.uniform(1.0, 35.0, n)
    lateral = rng.uniform(-28.0, 28.0, n)

    distance = np.hypot(forward, lateral)
    angle = xg._mouth_angle_from(forward, lateral)
    z = TRUE_INTERCEPT + TRUE_DISTANCE * distance + TRUE_ANGLE * angle
    p = 1.0 / (1.0 + np.exp(-z))

    return pd.DataFrame({
        "forward_m": forward,
        "lateral_m": lateral,
        "distance_m": distance,
        "angle_rad": angle,
        "true_p": p,
        "is_goal": rng.random(n) < p,
        "match_code": [f"2024-01-01-M{i % N_MATCHES:03d}" for i in range(n)],
    })


def as_xghub(df: pd.DataFrame, convention: str) -> pd.DataFrame:
    """The same shots, written the way xGHub records them."""
    if convention == "goal_centre":
        distance = df.distance_m
    else:
        distance = df.forward_m
    return pd.DataFrame({
        "distance_from_goal": distance,
        "angle_to_goal": np.degrees(np.arctan2(df.lateral_m, df.forward_m)),
        "angle_to_posts": np.degrees(df.angle_rad),
        "is_goal": df.is_goal,
        "match_code": df.match_code,
    })


def check_convention():
    print("1. The distance convention is inferred, not assumed\n")
    failures = 0
    for truth in xg.CONVENTIONS:
        shots = synthetic_shots(2000, seed=1)
        print(f"  dataset written as '{truth}':")
        got, residuals = xg.infer_distance_convention(as_xghub(shots, truth))
        ok = got == truth
        failures += not ok
        other = [c for c in xg.CONVENTIONS if c != truth][0]
        print(f"    inferred '{got}' -- {'ok' if ok else 'WRONG'}; "
              f"the other reading is out by "
              f"{residuals[other]:.1f} deg\n")
    return failures


def check_round_trip():
    print("2. Position -> distance and bearing -> position\n")
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(2000):
        x = rng.uniform(0.0, sg.PITCH_LENGTH_M)
        y = rng.uniform(0.0, sg.PITCH_WIDTH_M)
        goal = "right"
        feats = sg.shot_features(x, y, goal)
        if feats["forward_m"] <= 0.5:
            continue
        forward, lateral = xg.geometry_from(
            feats["distance_m"], feats["bearing_deg"], "goal_centre")
        worst = max(worst, float(abs(forward - feats["forward_m"])),
                    float(abs(lateral - feats["lateral_m"])))
    ok = worst < 1e-9
    print(f"   worst round-trip error {worst:.2e} m -- "
          f"{'ok' if ok else 'THE TWO DO NOT AGREE'}\n")
    return not ok


def check_recovery():
    print("3. Coefficients are recovered from shots drawn with them\n")
    shots = synthetic_shots()
    model = xg.fit(shots, source="synthetic", convention="goal_centre",
                   verbose=True)

    wanted = {"distance_m": TRUE_DISTANCE, "angle_rad": TRUE_ANGLE}
    print()
    failures = 0
    for name, got in zip(model.features, model.coefficients):
        want = wanted[name]
        err = abs(got - want) / abs(want)
        ok = err <= COEFFICIENT_TOLERANCE
        failures += not ok
        print(f"   {name:<12s} fitted {got:+.4f}  true {want:+.4f}  "
              f"off by {err:6.1%}  {'ok' if ok else 'OUT OF TOLERANCE'}")
    print(f"   {'intercept':<12s} fitted {model.intercept:+.4f}  "
          f"true {TRUE_INTERCEPT:+.4f}  "
          f"(off by {abs(model.intercept - TRUE_INTERCEPT):.3f})")

    # The intercept is judged through the predictions rather than on its own.
    # It is the least well determined coefficient, it trades off against the
    # slopes, and it sits near zero -- so a *relative* tolerance on it says
    # more about how close to zero it happens to be than about the fit. What
    # the model is for is probabilities, so those are what is checked.
    predicted = model.predict(distance_m=shots.distance_m.to_numpy(),
                              angle_rad=shots.angle_rad.to_numpy())
    rmse = float(np.sqrt(np.mean((predicted - shots.true_p.to_numpy()) ** 2)))
    worst = float(np.max(np.abs(predicted - shots.true_p.to_numpy())))
    ok = rmse <= PROBABILITY_RMSE_TOLERANCE
    failures += not ok
    print(f"\n   predicted probability vs the true one: RMSE {rmse:.4f}, "
          f"worst {worst:.4f}  {'ok' if ok else 'OUT OF TOLERANCE'}")

    # A model that cannot beat "every shot is the base rate" has learned
    # nothing, whatever its AUC looks like.
    beats_base = model.metrics["brier"] < model.metrics["brier_base"]
    print(f"\n   Brier {model.metrics['brier']:.4f} against a base-rate "
          f"floor of {model.metrics['brier_base']:.4f} -- "
          f"{'beats it' if beats_base else 'NO BETTER THAN THE BASE RATE'}")
    failures += not beats_base
    return failures


def check_penalties():
    """Penalties take the fixed value, whatever the geometry says."""
    print("\n4. Penalties bypass the fit\n")
    model = xg.fit(synthetic_shots(), verbose=False)
    failures = 0

    # The fitted model scored on penalty-spot geometry, for contrast.
    spot_distance = 11.0
    spot_angle = float(xg._mouth_angle_from(np.array(11.0), np.array(0.0)))
    fitted = float(model.predict(distance_m=np.array([spot_distance]),
                                 angle_rad=np.array([spot_angle]))[0])
    flagged = float(model.predict(distance_m=np.array([spot_distance]),
                                  angle_rad=np.array([spot_angle]),
                                  is_penalty=np.array([True]))[0])
    ok = flagged == model.penalty_xg
    failures += not ok
    print(f"   same shot, scored by geometry : {fitted:.3f}")
    print(f"   same shot, flagged a penalty  : {flagged:.3f}  "
          f"{'ok' if ok else 'DID NOT TAKE THE FIXED VALUE'}")

    # Geometry must not matter once flagged: a penalty is a penalty.
    far = float(model.predict(distance_m=np.array([40.0]),
                              angle_rad=np.array([0.05]),
                              is_penalty=np.array([True]))[0])
    ok = far == model.penalty_xg
    failures += not ok
    print(f"   flagged, absurd geometry      : {far:.3f}  "
          f"{'ok' if ok else 'GEOMETRY LEAKED IN'}")

    # Mixed arrays have to be element-wise, not all-or-nothing.
    mixed = model.predict(distance_m=np.array([11.0, 11.0, 25.0]),
                          angle_rad=np.array([spot_angle, spot_angle, 0.3]),
                          is_penalty=np.array([True, False, False]))
    ok = (mixed[0] == model.penalty_xg and mixed[1] != model.penalty_xg
          and mixed[2] != model.penalty_xg)
    failures += not ok
    print(f"   mixed array                   : "
          f"{np.round(mixed, 3).tolist()}  "
          f"{'ok' if ok else 'NOT ELEMENT-WISE'}")

    lo, hi = xg.PENALTY_XG_PLAUSIBLE
    ok = lo <= model.penalty_xg <= hi
    failures += not ok
    print(f"   the assumed value             : {model.penalty_xg:.2f}  "
          f"{'in' if ok else 'OUTSIDE'} the {lo}-{hi} range football produces")

    # A penalty recorded away from the spot is a detection error, and has to
    # be surfaced rather than handed a confident number.
    odd = model.penalties_out_of_place(
        distance_m=np.array([11.0, 10.2, 31.0, 25.0]),
        is_penalty=np.array([True, True, True, False]))
    ok = odd.tolist() == [2]
    failures += not ok
    print(f"   penalties away from the spot  : flagged index "
          f"{odd.tolist()}  {'ok' if ok else 'WRONG'}")
    return failures


def check_refusal():
    print("\n5. Too little data is refused rather than fitted\n")
    try:
        xg.fit(synthetic_shots(40, seed=9), verbose=False)
    except SystemExit as exc:
        print(f"   raised, as it should: {str(exc)[:70]}...")
        return 0
    print("   FITTED A MODEL on 40 shots")
    return 1


def main():
    failures = (check_convention() + check_round_trip() + check_recovery()
                + check_penalties() + check_refusal())
    if failures:
        raise SystemExit(f"\n{failures} check(s) failed")
    print("\n   All checks passed. The pipeline is ready for real shots.")


if __name__ == "__main__":
    main()
