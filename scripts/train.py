"""Train the classifiers (alive, team, three event detectors) on every public match and save
them under models/.

    python scripts/train.py [n_estimators=500]

Needs data_external/ (scripts/prepare_external.py) and the bootstrap cache (scripts/bootstrap.py).
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import games  # noqa: E402
from possession import matches, models, pipeline  # noqa: E402

CACHE = os.path.join(games.DATA_DIR, ".cache")
MODEL_DIR = os.path.join(ROOT, "models")


def main(**opts):
    t0 = time.time()
    ext = matches.external()
    tabs = pipeline.tables(ext, bank=pipeline.load_bank(CACHE))
    alive, team, names = pipeline.train(tabs, weighted=bool(int(opts.get("weighted", 1))),
                                        weight_power=float(opts.get("weight_power", 0.5)),
                                        n_estimators=int(opts.get("n_estimators", 500)))
    events = pipeline.train_events(tabs)
    models.save(alive, os.path.join(MODEL_DIR, "alive.txt"))
    models.save(team, os.path.join(MODEL_DIR, "team.txt"))
    for kind, booster in events.items():
        models.save(booster, os.path.join(MODEL_DIR, "event_{}.txt".format(kind)))
    with open(os.path.join(MODEL_DIR, "features.json"), "w") as f:
        json.dump(names, f, indent=0)
    print("trained on {} public matches (x2 with teams exchanged) in {:.0f}s".format(len(ext), time.time() - t0))


if __name__ == "__main__":
    main(**dict(a.split("=", 1) for a in sys.argv[1:]))
