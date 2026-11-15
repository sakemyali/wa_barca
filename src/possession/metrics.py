"""A fast, exactly equivalent implementation of the ranking metric, for tuning loops only.

The organisers' scorer (evaluation/possession_metrics.py) compares every predicted possession with
every true one. That is fine for one submission and too slow for a few thousand decoder settings.
Possessions inside a half are sorted and disjoint, so the overlapping pairs can be listed by a
linear sweep instead; everything after that (IoU, sort order, greedy matching, thresholds) is the
organisers' algorithm, line for line. tests/test_metrics.py checks the two agree to the last digit.

Every number quoted in the documentation comes from the organisers' scorer, never from here.
"""
import numpy as np

SPELL_IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def spells(codes, runs):
    """[(start, stop, code)] maximal H or A runs inside each scored span, in order."""
    out = []
    for start, stop in runs:
        w = codes[start:stop]
        if not len(w):
            continue
        breaks = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(w)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            k = int(w[a])
            if k < 2:
                out.append((start + int(a), start + int(b), k))
    return out


def spell_f1(gt_spells, pred_spells, ious=SPELL_IOUS):
    """Mean F1 over IoU thresholds of the one-to-one greedy possession matching."""
    overlaps = []
    j0 = 0
    for i, (gs, ge, gk) in enumerate(gt_spells):
        while j0 < len(pred_spells) and pred_spells[j0][1] <= gs:
            j0 += 1
        j = j0
        while j < len(pred_spells) and pred_spells[j][0] < ge:
            ps, pe, pk = pred_spells[j]
            if pk == gk and pe > gs:
                inter = min(ge, pe) - max(gs, ps)
                union = max(ge, pe) - min(gs, ps)
                overlaps.append((inter / float(union), i, j))
            j += 1
    overlaps.sort(key=lambda o: (-o[0], o[1], o[2]))
    n_gt, n_pred = len(gt_spells), len(pred_spells)
    f1s = []
    for thr in ious:
        used_gt, used_pred, tp = set(), set(), 0
        for iou, i, j in overlaps:
            if iou < thr:
                break
            if i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
            tp += 1
        fp, fn = n_pred - tp, n_gt - tp
        precision = tp / float(tp + fp) if tp + fp else 0.0
        recall = tp / float(tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s))


def macro_f1(gt, pred):
    """Mean F1 of H, A, D over frames where the truth is not X (a predicted X is a miss)."""
    f1s = []
    graded = gt != 3
    for k in (0, 1, 2):
        tp = float(np.sum((gt == k) & (pred == k)))
        fp = float(np.sum(graded & (gt != k) & (pred == k)))
        fn = float(np.sum((gt == k) & (pred != k)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))
