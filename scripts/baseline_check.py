"""Is the simulated tracking as hard as the real one? A label-free realism check.

The organisers publish what their baseline (nearest tracked player to the ball, D where the ball
track is missing, majority vote over a window) scores on the PRIVATE full-match ground truth:

    window   macroF1  spellF1
    raw      0.6532   0.0486
    +/-25    0.6752   0.1996
    +/-50    0.6724   0.2433
    +/-125   0.6356   0.2242

This script runs that same baseline on the public matches after degradation and prints the same
table. If the simulation is faithful the two tables should look alike; if the simulated matches
were too easy (or too hard) every number here would sit above (or below) the published one.

    python scripts/baseline_check.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import games  # noqa: E402
from possession import decode as dec  # noqa: E402
from possession import degrade, matches, pipeline  # noqa: E402
from possession.schema import D  # noqa: E402

PUBLISHED = {0: (0.6532, 0.0486), 25: (0.6752, 0.1996), 50: (0.6724, 0.2433), 125: (0.6356, 0.2242)}


def nearest_team(m):
    """0 home / 1 away / 2 where the ball is not tracked: step 1 of the organisers' baseline."""
    out = np.full(m.n_frames, D, np.int8)
    for s in range(0, m.n_frames, 20000):
        b = m.ball_xy[s:s + 20000]
        with np.errstate(invalid="ignore"):
            dh = np.sqrt(((m.home_xy[s:s + 20000] - b[:, None]) ** 2).sum(-1))
            da = np.sqrt(((m.away_xy[s:s + 20000] - b[:, None]) ** 2).sum(-1))
        dh = np.where(np.isfinite(dh), dh, np.inf).min(1)
        da = np.where(np.isfinite(da), da, np.inf).min(1)
        seen = np.isfinite(b[:, 0])
        out[s:s + 20000][seen] = np.where(dh[seen] <= da[seen], 0, 1)
    return out


def smooth(codes, blocks, window):
    """Step 2: centred majority vote between home and away; D only where neither was seen."""
    if not window:
        return codes
    out = codes.copy()
    for _, s, e in blocks:
        seg = codes[s:e]
        ch = np.concatenate(([0], np.cumsum(seg == 0)))
        ca = np.concatenate(([0], np.cumsum(seg == 1)))
        idx = np.arange(len(seg))
        lo, hi = np.maximum(0, idx - window), np.minimum(len(seg), idx + window + 1)
        h, a = ch[hi] - ch[lo], ca[hi] - ca[lo]
        out[s:e] = np.where(h + a == 0, D, np.where(h >= a, 0, 1))
    return out


def main():
    bank = pipeline.load_bank(os.path.join(games.DATA_DIR, ".cache"))
    rows = {w: [] for w in PUBLISHED}
    for i, m in enumerate(matches.external()):
        d = degrade.degrade(m, bank, seed=2 * i)
        codes = nearest_team(d)
        blocks = d.blocks()
        table = {"label": np.concatenate([m.label[s:e] for _, s, e in blocks]),
                 "period": np.concatenate([m.period[s:e] for _, s, e in blocks])}
        gt, cut = pipeline.table_gt(table)
        for w in PUBLISHED:
            pred = np.concatenate([smooth(codes, blocks, w)[s:e] for _, s, e in blocks])
            r = pipeline.score(gt, pipeline.with_gap(pred, cut))
            rows[w].append((r["macro_f1"], r["spell_f1"]))
    print("window   simulated public matches (mean of 10)    published, private full match")
    print("         macroF1   spellF1                         macroF1   spellF1")
    for w, vals in rows.items():
        v = np.mean(vals, 0)
        print("{:<8} {:.4f}    {:.4f}                          {:.4f}    {:.4f}".format(
            "raw" if not w else "+/-{}".format(w), v[0], v[1], *PUBLISHED[w]))


if __name__ == "__main__":
    main()
