"""The learned parts: gradient-boosted trees over the engineered features.

Two classifiers, trained only on public matches (other games), after those matches have been
degraded to look like the provided tracking (degrade.py):

  alive   P(dead ball | frame). Sees the players' collective movement, the visibility pattern of
          the ball track, and where a visible ball is relative to the lines. It does NOT see fine
          ball kinematics: during a stoppage the public sources have no real ball, so anything
          learned from it there would be an artefact of the simulation.
  team    P(home has it | frame, ball alive). Trained on live frames only. Sees everything.

Both are made exactly indifferent to which side is called home by training on every match twice,
once as recorded and once with the teams exchanged.
"""
import os

import lightgbm as lgb
import numpy as np

from . import features
from .schema import D, H

# Head-counts are excluded everywhere: public tracking always has 22 players and the provided
# tracking does not, so a count would be a pure source detector.
_COUNT = ("p:h_n", "p:a_n", "p:all_n")
ALIVE_BALL = ("b:b_out", "b:bx_edge", "b:by_edge")

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                  n_estimators=500, verbose=-1, seed=0, deterministic=True, force_row_wise=True,
                  n_jobs=8)


def is_count(name):
    return name.startswith(_COUNT)


def alive_columns(names):
    return [i for i, n in enumerate(names)
            if not is_count(n) and (n.startswith(("p:", "v:")) or n in ALIVE_BALL)]


def players_columns(names):
    return [i for i, n in enumerate(names) if n.startswith("p:") and not is_count(n)]


def team_columns(names):
    return [i for i, n in enumerate(names) if not is_count(n)]


def spell_weights(label, power=0.5):
    """(length of the constant-label run a frame sits in) ** -power, normalised to mean 1.

    The metric counts each possession once, however long. Unweighted training (power 0) lets a
    one-minute possession outvote sixty one-second ones; power 1 gives every possession the same
    total say, which is the metric's own view; 0.5 sits between.
    """
    change = np.flatnonzero(label[1:] != label[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(label)]))
    w = np.empty(len(label), np.float32)
    for a, b in zip(bounds[:-1], bounds[1:]):
        w[a:b] = float(b - a) ** -power
    return w / w.mean()


def fit(X, y, weight=None, **overrides):
    params = dict(LGB_PARAMS, **overrides)
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y, sample_weight=weight)
    return model.booster_


def predict(booster, X):
    return booster.predict(X, num_threads=8)


def save(booster, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    booster.save_model(path)


def load(path):
    return lgb.Booster(model_file=path)
