"""Where the possessions are lost: recall and precision by possession length, and what a perfect
alive/dead call or a perfect team call would be worth. Held-out public matches only.

    python scripts/error_analysis.py [tag]
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
from tune_decoder import load_oof  # noqa: E402

BINS = [(0, 25), (25, 50), (50, 100), (100, 250), (250, 10 ** 9)]


def matched(gt_spells, pred_spells, thr):
    """Indices of matched ground-truth and predicted possessions at one IoU threshold."""
    ov, j0 = [], 0
    for i, (gs, ge, gk) in enumerate(gt_spells):
        while j0 < len(pred_spells) and pred_spells[j0][1] <= gs:
            j0 += 1
        j = j0
        while j < len(pred_spells) and pred_spells[j][0] < ge:
            ps, pe, pk = pred_spells[j]
            if pk == gk and pe > gs:
                ov.append(((min(ge, pe) - max(gs, ps)) / float(max(ge, pe) - min(gs, ps)), i, j))
            j += 1
    ov.sort(key=lambda o: (-o[0], o[1], o[2]))
    ug, up = set(), set()
    for iou, i, j in ov:
        if iou < thr:
            break
        if i in ug or j in up:
            continue
        ug.add(i)
        up.add(j)
    return ug, up


def main(tag="v3"):
    oof = load_oof(tag)
    with open(os.path.join(ROOT, "experiments", "decoder_{}.json".format(tag))) as f:
        params = dec.DecoderParams(*json.load(f)["params"])
    tot = {b: [0, 0, 0, 0] for b in BINS}
    res = {"as predicted": [], "perfect alive/dead": [], "perfect team": []}
    by_iou = {t: [] for t in metrics.SPELL_IOUS}
    for name, (p_dead, p_home, lab, per, p_ev) in sorted(oof.items()):
        blocks = pipeline.blocks_of(per)
        runs = [(s, e) for _, s, e in blocks]
        gts = metrics.spells(lab, runs)
        ev = None if p_ev is None else p_ev.astype(float)
        pred = dec.decode(p_dead.astype(float), p_home.astype(float), blocks, params, p_events=ev)
        ps = metrics.spells(pred, runs)
        res["as predicted"].append(metrics.spell_f1(gts, ps))
        for t in by_iou:
            by_iou[t].append(metrics.spell_f1(gts, ps, (t,)))
        ug, up = matched(gts, ps, 0.5)
        for i, (s, e, _) in enumerate(gts):
            for b in BINS:
                if b[0] <= e - s < b[1]:
                    tot[b][0] += 1
                    tot[b][1] += i in ug
        for j, (s, e, _) in enumerate(ps):
            for b in BINS:
                if b[0] <= e - s < b[1]:
                    tot[b][2] += 1
                    tot[b][3] += j in up
        o1 = dec.decode(np.where(lab == 2, 1 - 1e-6, 1e-6), p_home.astype(float), blocks, params, p_events=ev)
        res["perfect alive/dead"].append(metrics.spell_f1(gts, metrics.spells(o1, runs)))
        o2 = dec.decode(p_dead.astype(float), np.where(lab == 0, 1 - 1e-6, np.where(lab == 1, 1e-6, 0.5)), blocks,
                        params, p_events=ev)
        res["perfect team"].append(metrics.spell_f1(gts, metrics.spells(o2, runs)))
    print("decoder", [round(v, 2) for v in params])
    for k, v in res.items():
        print("  {:<20} spellF1 {:.4f}".format(k, np.mean(v)))
    print("  F1 by IoU threshold:", {t: round(float(np.mean(v)), 3) for t, v in by_iou.items()})
    print("  possession length   true: n, recall@0.5 | predicted: n, precision@0.5")
    for b in BINS:
        g, gm, p, pm_ = tot[b]
        print("  {:>4.0f}-{:<6} s        {:>5} {:.3f}         | {:>5} {:.3f}".format(
            b[0] / 25, "{:.0f}".format(b[1] / 25) if b[1] < 10 ** 9 else "", g, gm / max(g, 1), p, pm_ / max(p, 1)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v3")
