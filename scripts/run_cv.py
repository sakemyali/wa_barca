"""Cross-validate the pipeline on the public matches (held-out matches, degraded like the
provided one) and save out-of-fold probabilities for decoder tuning.

    python scripts/run_cv.py [tag] [key=value ...]

Writes experiments/oof_<tag>.npz and appends one line to experiments/results.jsonl.
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

import games  # noqa: E402
from possession import decode as dec  # noqa: E402
from possession import matches, pipeline  # noqa: E402

CACHE = os.path.join(games.DATA_DIR, ".cache")
# Three folds, each holding out one Metrica sample game and two or three DFL matches.
FOLDS = [
    ["dfl_J03WMX", "dfl_J03WOY", "dfl_J03WR9", "metrica_sample_1"],
    ["dfl_J03WN1", "dfl_J03WPY", "metrica_sample_2"],
    ["dfl_J03WOH", "dfl_J03WQQ", "metrica_sample_3"],
]


def main(tag="v1", **opts):
    t0 = time.time()
    weighted = bool(int(opts.get("weighted", 1)))
    n_estimators = int(opts.get("n_estimators", 300))
    with_events = bool(int(opts.get("events", 1)))
    weight_power = float(opts.get("weight_power", 0.5))
    extra = {k: (float(v) if "." in v else int(v)) for k, v in opts.items()
             if k in ("num_leaves", "learning_rate", "min_child_samples", "colsample_bytree")}
    ext = matches.external()
    bank = pipeline.load_bank(CACHE)
    tabs = pipeline.tables(ext, bank=bank)
    by_name = {t["name"]: t for t in tabs}
    print("features: {} tables x {} columns in {:.0f}s".format(len(tabs), len(tabs[0]["names"]), time.time() - t0), flush=True)

    oof = {}
    for k, held in enumerate(FOLDS):
        train = [t for t in tabs if t["name"].replace("_swap", "") not in held]
        alive, team, _ = pipeline.train(train, weighted=weighted, weight_power=weight_power,
                                        n_estimators=n_estimators, **extra)
        events = pipeline.train_events(train) if with_events else None
        for name in held:
            t = by_name[name]
            p_dead, p_home = pipeline.infer(t, alive, team)
            p_ev = pipeline.infer_events(t, events) if with_events else np.zeros((len(p_dead), 3))
            oof[name] = (p_dead.astype(np.float32), p_home.astype(np.float32), t["label"], t["period"],
                         p_ev.astype(np.float32))
            lab = t["label"]
            al = lab < 2
            print("  fold {} {:<18} dead acc {:.4f} | team acc on live frames {:.4f}".format(
                k, name, ((p_dead > 0.5) == (lab == 2)).mean(), ((p_home[al] > 0.5) == (lab[al] == 0)).mean()), flush=True)
    os.makedirs(os.path.join(ROOT, "experiments"), exist_ok=True)
    np.savez_compressed(os.path.join(ROOT, "experiments", "oof_{}.npz".format(tag)),
                        **{"{}|{}".format(n, f): v for n, vals in oof.items()
                           for f, v in zip(("p_dead", "p_home", "label", "period", "p_events"), vals)})

    # a first decode with default parameters, so every CV run reports the ranking metric
    params = dec.DecoderParams()
    rows = []
    for name, (p_dead, p_home, lab, per, _) in sorted(oof.items()):
        pred = dec.decode(p_dead, p_home, pipeline.blocks_of(per), params)
        gt, cut = pipeline.table_gt({"label": lab, "period": per})
        r = pipeline.score(gt, pipeline.with_gap(pred, cut))
        rows.append(r)
        print("  {:<18} spellF1 {:.4f} macroF1 {:.4f} spells {} / {}".format(
            name, r["spell_f1"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]), flush=True)
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("spell_f1", "macro_f1", "alive_dead_f1", "team_acc")}
    print("MEAN (default decoder): spellF1 {spell_f1:.4f} macroF1 {macro_f1:.4f} alive/dead F1 {alive_dead_f1:.4f} "
          "team acc {team_acc:.4f}".format(**mean))
    with open(os.path.join(ROOT, "experiments", "results.jsonl"), "a") as f:
        f.write(json.dumps({"tag": tag, "opts": opts, "decoder": list(params), "seconds": round(time.time() - t0),
                            "mean": mean, "time": time.strftime("%Y-%m-%d %H:%M")}) + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    tag = args[0] if args and "=" not in args[0] else "v1"
    main(tag, **dict(a.split("=", 1) for a in args if "=" in a))
