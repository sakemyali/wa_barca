"""Choose the decoder's four numbers on held-out public matches, by the ranking metric.

    python scripts/tune_decoder.py [tag]

Reads experiments/oof_<tag>.npz (out-of-fold probabilities from run_cv.py), decodes every match
under a grid of settings and writes the best to experiments/decoder_<tag>.json. The provided
match's public ground truth plays no part in this.
"""
import itertools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from possession import decode as dec  # noqa: E402
from possession import metrics, pipeline  # noqa: E402

GRID = {
    "temper": (0.1, 0.2, 0.3),
    "cost_team": (1.0, 2.0, 3.0, 4.5, 6.5),
    "cost_stop": (2.0, 6.0, 14.0),
    "cost_restart": (2.0, 6.0, 14.0),
    "gain_team": (0.0, 0.5, 1.0, 2.0),
    "gain_dead": (0.0, 1.0),            # used for both the stop and the restart detector
}


def load_oof(tag):
    z = np.load(os.path.join(ROOT, "experiments", "oof_{}.npz".format(tag)))
    names = sorted({k.split("|")[0] for k in z.files})
    out = {}
    for n in names:
        vals = [z["{}|{}".format(n, f)] for f in ("p_dead", "p_home", "label", "period")]
        key = "{}|p_events".format(n)
        vals.append(z[key] if key in z.files and z[key].any() else None)
        out[n] = tuple(vals)
    return out


def grid(with_events):
    out = []
    for t, ct, cs, cr, gt, gd in itertools.product(*GRID.values()):
        if not with_events and (gt or gd):
            continue
        out.append(dec.DecoderParams(t, ct, cs, cr, gt, gd, gd))
    return out


def _one_match(args):
    (p_dead, p_home, lab, per, p_ev), params_list = args
    blocks = pipeline.blocks_of(per)
    paths = dec.decode_batch(p_dead.astype(np.float64), p_home.astype(np.float64), blocks, params_list,
                             p_events=None if p_ev is None else p_ev.astype(np.float64))
    runs = [(s, e) for _, s, e in blocks]
    gt_spells = metrics.spells(lab, runs)
    spell = [metrics.spell_f1(gt_spells, metrics.spells(path, runs)) for path in paths]
    macro = [metrics.macro_f1(lab, path) for path in paths]
    return spell, macro


def evaluate_grid(oof, params_list, n_jobs=5):
    """spellF1 and macroF1 per (match, setting): two arrays [n_matches, n_settings]."""
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=n_jobs)(delayed(_one_match)((v, params_list)) for _, v in sorted(oof.items()))
    return np.array([r[0] for r in res]), np.array([r[1] for r in res])


def main(tag="v1"):
    t0 = time.time()
    oof = load_oof(tag)
    with_events = all(v[4] is not None for v in oof.values())
    params_list = grid(with_events)
    spell, macro = evaluate_grid(oof, params_list)
    mean = spell.mean(0)
    order = np.argsort(-mean)
    print("{} settings x {} matches in {:.0f}s (event detectors: {})".format(
        len(params_list), len(oof), time.time() - t0, "on" if with_events else "off"))
    print("rank  temper team stop restart g_team g_dead | spellF1 mean (min..max over matches) | macroF1")
    for r, i in enumerate(order[:12]):
        p = params_list[i]
        print("{:>4}  {:>5} {:>5} {:>4} {:>7} {:>6} {:>6} | {:.4f} ({:.3f}..{:.3f}) | {:.4f}".format(
            r + 1, p.temper, p.cost_team, p.cost_stop, p.cost_restart, p.gain_team, p.gain_stop, mean[i],
            spell[:, i].min(), spell[:, i].max(), macro[:, i].mean()))
    if with_events:
        base = [i for i, p in enumerate(params_list) if not (p.gain_team or p.gain_stop)]
        b = base[int(np.argmax(mean[base]))]
        print("best without event detectors: {:.4f} at {}".format(mean[b], list(params_list[b])[:4]))
    best = params_list[int(order[0])]
    names = sorted(oof)
    print("per match at the best setting:", {n: round(float(spell[k, order[0]]), 3) for k, n in enumerate(names)})
    with open(os.path.join(ROOT, "experiments", "decoder_{}.json".format(tag)), "w") as f:
        json.dump({"params": list(best), "spell_f1": float(mean[order[0]]), "macro_f1": float(macro[:, order[0]].mean()),
                   "per_match": {n: float(spell[k, order[0]]) for k, n in enumerate(names)}}, f, indent=1)
    return best


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")
