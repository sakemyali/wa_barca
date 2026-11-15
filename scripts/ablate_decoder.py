"""How much each way of turning probabilities into possessions is worth, on the same held-out
probabilities: per-frame argmax, the organisers' majority-vote smoothing, Viterbi, and Viterbi with
event-aware transition costs.

    python scripts/ablate_decoder.py [tag]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from possession import decode as dec  # noqa: E402
from possession import metrics, pipeline  # noqa: E402
from tune_decoder import evaluate_grid, grid, load_oof  # noqa: E402


def main(tag="v3"):
    oof = load_oof(tag)
    rows = []

    def add(name, fn):
        s, m, n = [], [], []
        for _, (p_dead, p_home, lab, per, p_ev) in sorted(oof.items()):
            blocks = pipeline.blocks_of(per)
            runs = [(a, b) for _, a, b in blocks]
            pred = fn(p_dead.astype(float), p_home.astype(float), blocks, p_ev)
            ps = metrics.spells(pred, runs)
            s.append(metrics.spell_f1(metrics.spells(lab, runs), ps))
            m.append(metrics.macro_f1(lab, pred))
            n.append(len(ps) / max(len(metrics.spells(lab, runs)), 1))
        rows.append((name, np.mean(s), np.mean(m), np.mean(n)))

    def argmax(p_dead, p_home, blocks, p_ev):
        return dec.emissions(p_dead, p_home).argmax(1).astype(np.int8)

    add("per-frame argmax", argmax)
    for w in (12, 25, 50, 100):
        add("argmax + majority vote +/-{}".format(w),
            lambda pd, ph, b, ev, w=w: dec.majority_smooth(argmax(pd, ph, b, ev), b, w))

    for label, with_events in (("Viterbi, tuned", False), ("Viterbi + event-aware costs, tuned", True)):
        params = [p for p in grid(True) if with_events or not (p.gain_team or p.gain_stop)]
        spell, macro = evaluate_grid(oof, params)
        best = params[int(np.argmax(spell.mean(0)))]
        add("{} {}".format(label, [round(v, 1) for v in best]),
            lambda pd, ph, b, ev, best=best: dec.decode(pd, ph, b, best, p_events=None if ev is None else ev.astype(float)))

    print("{:<64} spellF1  macroF1  possessions predicted / true".format("decoder ({} held-out probabilities)".format(tag)))
    for name, s, m, n in rows:
        print("{:<64} {:.4f}   {:.4f}   {:.2f}".format(name, s, m, n))
    with open(os.path.join(ROOT, "experiments", "decoder_ablation_{}.json".format(tag)), "w") as f:
        json.dump([{"decoder": r[0], "spell_f1": r[1], "macro_f1": r[2], "spell_ratio": r[3]} for r in rows], f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v3")
