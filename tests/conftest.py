import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (os.path.join(ROOT, "src"), ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from possession import schema  # noqa: E402


@pytest.fixture
def toy_match():
    """Two short halves of smooth synthetic movement, with labels. No real data involved."""
    rng = np.random.default_rng(7)
    n, per_half = 1600, 800
    t = np.arange(n)[:, None]
    def team(base_x, sign):
        x = base_x + 8 * np.sin(t / 90.0 + rng.uniform(0, 6, 11)) + rng.normal(0, 6, 11)
        y = np.linspace(6, 62, 11)[None, :] + 4 * np.cos(t / 70.0 + rng.uniform(0, 6, 11))
        return np.stack([x, y], 2).astype(np.float32)
    home, away = team(35.0, 1), team(70.0, -1)
    # second half: the teams change ends
    home[per_half:] = np.array([105.0, 68.0], np.float32) - home[per_half:]
    away[per_half:] = np.array([105.0, 68.0], np.float32) - away[per_half:]
    ball = np.stack([52.5 + 30 * np.sin(np.arange(n) / 40.0), 34 + 20 * np.cos(np.arange(n) / 55.0)], 1).astype(np.float32)
    label = np.zeros(n, np.int8)
    for s in range(0, n, 200):
        label[s + 60:s + 120] = 1
        label[s + 150:s + 200] = 2
    period = np.r_[np.ones(per_half, np.int8), np.full(n - per_half, 2, np.int8)]
    return schema.Match("toy", (105.0, 68.0), ball, home, away, label, period, np.full(n, np.nan, np.float32))
