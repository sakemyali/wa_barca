"""The fast tuning metric must equal the organisers' scorer exactly."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation"))
import possession_metrics as pm  # noqa: E402

from possession import metrics  # noqa: E402


def _random_labels(rng, n, mean_run):
    out = np.empty(n, np.int8)
    i = 0
    while i < n:
        length = 1 + rng.geometric(1.0 / mean_run)
        out[i:i + length] = rng.integers(0, 3)
        i += length
    return out


def test_fast_metrics_match_the_official_scorer():
    rng = np.random.default_rng(0)
    for trial in range(12):
        n = 6000
        gt = _random_labels(rng, n, 80)
        gt[:200] = 3
        gt[2800:3100] = 3            # a halftime hole
        gt[-150:] = 3
        pred = gt.copy()
        flip = rng.random(n) < 0.02 * (trial + 1)
        pred[flip] = rng.integers(0, 3, flip.sum())
        if trial % 3 == 0:
            pred = _random_labels(rng, n, 40)
        official = pm.evaluate(gt, pred)
        runs = pm.valid_runs(gt)
        fast = metrics.spell_f1(metrics.spells(gt, runs), metrics.spells(pred, runs))
        assert abs(fast - official["spell_f1_mean"]) < 1e-12
        assert abs(metrics.macro_f1(gt, pred) - official["macro_f1"]) < 1e-12
