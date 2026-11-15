"""Step 1-2 of the pipeline: a label-free first guess at the provided match's stoppages (from
player movement only), and the ball-visibility bank cut along it. Writes <data>/.cache/bootstrap.npz.

    python scripts/bootstrap.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import games  # noqa: E402
from possession import degrade, matches, models, pipeline  # noqa: E402
from possession.schema import D, X  # noqa: E402

CACHE = os.path.join(games.DATA_DIR, ".cache")


def main():
    t0 = time.time()
    ext = matches.external()
    tabs = pipeline.tables(ext, bank=None, ball_sigma=degrade.BALL_SIGMA)
    names = tabs[0]["names"]
    cols = models.players_columns(names)
    Xp, yp, _ = pipeline.stack(tabs, cols, lambda t: t["label"] != X)
    players_only = models.fit(Xp, (yp == D).astype(np.int8))
    print("players-only alive model: {} rows, {} features, {:.0f}s".format(len(yp), len(cols), time.time() - t0))

    prov = matches.provided()
    tab = pipeline.tables([prov], bank=None, swaps=(False,))[0]
    p_players = models.predict(players_only, tab["X"][:, cols].astype(np.float32))
    blocks = pipeline.blocks_of(tab["period"])
    missing = ~np.isfinite(prov.ball_xy[tab["rows"], 0])
    # Stoppages from player movement ALONE. The ball track is deliberately not consulted: the
    # bank exists to measure how visibility relates to stoppages, so the stoppages must come from
    # a view that shares nothing with it. (Fitting both jointly by EM collapses to "missing =
    # dead", which is circular and wrong on exactly the frames that matter.)
    dead = pipeline.hard_dead(p_players, blocks)
    print("estimated dead share {:.3f}; dead segments {}".format(dead.mean(), len(degrade.runs(dead))))

    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, "bootstrap.npz"), rows=tab["rows"], period=tab["period"],
                        p_players=p_players.astype(np.float32), dead=dead, missing=missing)
    models.save(players_only, os.path.join(ROOT, "models", "alive_players_only.txt"))
    bank = degrade.visibility_bank(~missing, dead, blocks)
    for key in ("dead", "alive"):
        m = bank[key]
        frac = np.array([x.mean() for x in m])
        lens = np.array([len(x) for x in m])
        print("bank[{}]: {} segments, median {:.0f} frames | visible share: all-missing {:.2f} of segments, "
              "all-visible {:.2f}, frame-weighted visible {:.3f}".format(
                  key, len(m), np.median(lens), (frac == 0).mean(), (frac == 1).mean(),
                  sum(x.sum() for x in m) / lens.sum()))
    print("done in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
