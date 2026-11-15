"""Label-free transfer check ("adversarial validation"): can a classifier tell a frame of the
provided match from a frame of a degraded public match? The features it leans on are the ones
whose distribution differs between the two, i.e. where a model trained on one may misbehave on
the other.

    python scripts/domain_check.py
"""
import os
import sys

import lightgbm as lgb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import games  # noqa: E402
from possession import matches, models, pipeline  # noqa: E402


def main(n_each=120000, seed=0):
    rng = np.random.default_rng(seed)
    bank = pipeline.load_bank(os.path.join(games.DATA_DIR, ".cache"))
    ext = matches.external()
    ext_tabs = pipeline.tables([ext[1], ext[5], ext[7], ext[9]], bank=bank, swaps=(False,))
    prov = pipeline.tables([matches.provided()], bank=None, swaps=(False,))[0]
    names = prov["names"]
    Xe = np.concatenate([t["X"] for t in ext_tabs])
    Xe = Xe[rng.choice(len(Xe), n_each, replace=False)].astype(np.float32)
    Xp = prov["X"][rng.choice(len(prov["X"]), n_each, replace=False)].astype(np.float32)
    X = np.concatenate([Xe, Xp])
    y = np.r_[np.zeros(n_each), np.ones(n_each)]
    idx = rng.permutation(len(y))
    tr, te = idx[:len(y) * 2 // 3], idx[len(y) * 2 // 3:]
    for label, cols in (("alive-model features", models.alive_columns(names)),
                        ("team-model features", models.team_columns(names))):
        m = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.5, verbose=-1, n_jobs=4, seed=0)
        m.fit(X[tr][:, cols], y[tr])
        p = m.predict_proba(X[te][:, cols])[:, 1]
        order = np.argsort(p)
        rank = np.empty(len(p))
        rank[order] = np.arange(len(p))
        pos = y[te] == 1
        auc = (rank[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / (pos.sum() * (~pos).sum())
        gain = m.booster_.feature_importance("gain")
        top = sorted(zip(gain / gain.sum(), [names[c] for c in cols]), reverse=True)[:14]
        print("{}: provided-vs-public AUC {:.3f} (0.5 = indistinguishable)".format(label, auc))
        print("   most separating:", ", ".join("{} {:.0%}".format(n, g) for g, n in top))


if __name__ == "__main__":
    main()
