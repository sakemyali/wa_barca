"""End-to-end steps, shared by training, cross-validation and inference.

The order things happen in, for the final submission:

  1. bootstrap   a players-only alive/dead model (public matches) is run on the provided match
                 and decoded into segments. It never looks at the ball. No label of the provided
                 match is involved. The result is a first guess at where its stoppages are.
  2. bank        the provided ball track's visible / missing pattern is cut along that guess.
  3. degrade     every public match gets that pattern pasted on, plus smoothing and track loss.
  4. train       alive and team classifiers on the degraded public matches.
  5. infer       both classifiers on the provided match.
  6. decode      Viterbi over H / A / D per half.

"""
import os
import sys

import numpy as np
from joblib import Parallel, delayed

from . import decode as dec
from . import degrade, features, kickoff, models
from .schema import D, H, X

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402  (the organisers' scorer, vendored unchanged)

TRAIN_STRIDE = 3          # keep every third frame for training; neighbours are near-duplicates
N_JOBS = 4                # feature building is memory-bound, not CPU-bound


# ----------------------------------------------------------------------------------------------
# feature tables
# ----------------------------------------------------------------------------------------------

def _table(match, bank, seed, swap, ball_sigma):
    """Features + labels of one (optionally degraded, optionally team-swapped) match."""
    m = match
    if bank is not None:
        m = degrade.degrade(m, bank, seed)
        ball_sigma = 0.0                       # degrade() already smoothed the ball
    if swap:
        m = features.swap_teams(m)
    names, Xf, rows = features.build(m, ball_sigma=ball_sigma)
    return {"name": m.name, "names": names, "X": Xf.astype(np.float16), "rows": rows,
            "label": m.label[rows], "period": m.period[rows]}


def tables(matches, bank=None, swaps=(False, True), base_seed=0, ball_sigma=0.0):
    """Feature tables for a list of matches, in parallel. Seeds are fixed per (match, swap)."""
    jobs = [(m, bank, base_seed + 2 * i + int(sw), sw, ball_sigma)
            for i, m in enumerate(matches) for sw in swaps]
    return Parallel(n_jobs=N_JOBS)(delayed(_table)(*j) for j in jobs)


def stack(tabs, cols, keep, stride=TRAIN_STRIDE, weight_power=0.5):
    """Training matrix from several tables. `keep(table)` -> bool mask of usable frames."""
    Xs, ys, ws = [], [], []
    for t in tabs:
        mask = keep(t)
        idx = np.flatnonzero(mask)[::stride]
        Xs.append(t["X"][idx][:, cols].astype(np.float32))
        ys.append(t["label"][idx])
        ws.append(models.spell_weights(t["label"], weight_power)[idx])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


# ----------------------------------------------------------------------------------------------
# training and inference
# ----------------------------------------------------------------------------------------------

def train(tabs, weighted=True, weight_power=0.5, **lgb_overrides):
    """(alive booster, team booster, feature names) from degraded public tables."""
    names = tabs[0]["names"]
    ca, ct = models.alive_columns(names), models.team_columns(names)
    Xa, ya, wa = stack(tabs, ca, lambda t: t["label"] != X, weight_power=weight_power)
    alive = models.fit(Xa, (ya == D).astype(np.int8), wa if weighted else None, **lgb_overrides)
    del Xa
    Xt, yt, wt = stack(tabs, ct, lambda t: t["label"] < D, weight_power=weight_power)
    team = models.fit(Xt, (yt == H).astype(np.int8), wt if weighted else None, **lgb_overrides)
    return alive, team, names


def infer(table, alive, team):
    """(p_dead, p_home) for every row of one feature table."""
    names = table["names"]
    Xf = table["X"].astype(np.float32)
    p_dead = models.predict(alive, Xf[:, models.alive_columns(names)])
    p_home = models.predict(team, Xf[:, models.team_columns(names)])
    return p_dead, p_home


EVENT_HALF_WIDTH = 2      # an event's positive frames: the change itself and two either side
EVENTS = ("team", "stop", "restart")


def event_targets(label, period):
    """Bool [n, 3]: is there a team change / a stoppage start / a restart within two frames?"""
    prev, cur = label[:-1], label[1:]
    same_half = period[:-1] == period[1:]
    kinds = [(prev < D) & (cur < D) & (prev != cur), (prev < D) & (cur == D), (prev == D) & (cur < D)]
    out = np.zeros((len(label), 3), bool)
    w = EVENT_HALF_WIDTH
    for k, hit in enumerate(kinds):
        for t in np.flatnonzero(hit & same_half) + 1:
            out[max(t - w, 0):t + w + 1, k] = True
    return out


def train_events(tabs, **lgb_overrides):
    """Three detectors of the instant something changes. The team-change detector sees every
    feature; the stop and restart detectors see what the alive model sees, for the same reason."""
    names = tabs[0]["names"]
    cols = {"team": models.team_columns(names), "stop": models.alive_columns(names),
            "restart": models.alive_columns(names)}
    boosters = {}
    for k, kind in enumerate(EVENTS):
        Xs, ys = [], []
        for t in tabs:
            y = event_targets(t["label"], t["period"])[:, k]
            usable = (t["label"] < D) if kind == "team" else (t["label"] != X)
            near = np.convolve(y, np.ones(13), "same") > 0          # keep every frame near an event
            keep = usable & (near | (np.arange(len(y)) % 4 == 0))
            Xs.append(t["X"][keep][:, cols[kind]].astype(np.float32))
            ys.append(y[keep])
        params = dict(n_estimators=200, **lgb_overrides)
        boosters[kind] = models.fit(np.concatenate(Xs), np.concatenate(ys).astype(np.int8), **params)
    return boosters


def infer_events(table, boosters):
    """Detector probabilities [n, 3] in the order of EVENTS."""
    names = table["names"]
    Xf = table["X"].astype(np.float32)
    cols = {"team": models.team_columns(names), "stop": models.alive_columns(names),
            "restart": models.alive_columns(names)}
    return np.stack([models.predict(boosters[k], Xf[:, cols[k]]) for k in EVENTS], 1)


def blocks_of(period):
    """[(period, start, stop)] over a table's rows (rows of the two halves are concatenated)."""
    out = []
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        if len(idx):
            out.append((p, int(idx[0]), int(idx[-1]) + 1))
    return out


def before_kickoff(match, table, p_dead):
    """Bool mask over a table's rows: frames of a half that precede its opening kickoff.

    Only applied when the half does not open in play: the detector's answer is used if the alive
    model itself calls the first five seconds a dead ball. A half that is already in play at its
    first frame (the usual case, and every public match) is left entirely to the models.
    """
    out = np.zeros(len(p_dead), bool)
    for (_, start, stop), (_, r0, r1) in zip(match.blocks(), blocks_of(table["period"])):
        k = kickoff.opening_kickoff(match.home_xy[start:stop], match.away_xy[start:stop], match.pitch)
        if k > 0 and p_dead[r0:r0 + 125].mean() > 0.5:
            out[r0:r0 + k] = True
    return out


# ----------------------------------------------------------------------------------------------
# bootstrap: a label-free first guess at the provided match's stoppages
# ----------------------------------------------------------------------------------------------

def hard_dead(p_dead, blocks, cost=8.0, temper=0.3):
    """Viterbi alive/dead segmentation of a posterior, so the bank is cut along clean segments."""
    out = np.zeros(len(p_dead), bool)
    p = np.clip(p_dead, 1e-6, 1 - 1e-6)
    for _, s, e in blocks:
        emit = np.stack([np.log1p(-p[s:e]), np.full(e - s, -1e9), np.log(p[s:e])], 1)   # H = alive, A unused
        path = dec.viterbi(emit, dec.DecoderParams(temper, 1e9, cost, cost))
        out[s:e] = path == D
    return out


# ----------------------------------------------------------------------------------------------
# scoring and output
# ----------------------------------------------------------------------------------------------

def score(gt, pred):
    """The organisers' metrics for a pair of code arrays (same length)."""
    r = pm.evaluate(np.asarray(gt, np.int8), np.asarray(pred, np.int8))
    return {"spell_f1": r["spell_f1_mean"], "macro_f1": r["macro_f1"], "alive_dead_f1": r["alive_dead_f1"],
            "team_acc": r["team_acc_both_alive"], "n_gt_spells": r["n_gt_spells"],
            "n_pred_spells": r["n_pred_spells"], "f1_D": r["per_class"]["D"]["f1"],
            "trans_f1_25": r["transitions"][25]["f1"]}


def table_gt(table):
    """Ground-truth codes of a table as a full-length array the scorer accepts: the two halves
    separated by one X frame so that nothing is scored across halftime."""
    lab, per = table["label"], table["period"]
    cut = int(np.flatnonzero(per == 2)[0]) if (per == 2).any() else len(lab)
    return np.concatenate([lab[:cut], [X], lab[cut:]]).astype(np.int8), cut


def with_gap(pred, cut):
    return np.concatenate([pred[:cut], [X], pred[cut:]]).astype(np.int8)


def write_submission(path, codes):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    letters = np.array(["H", "A", "D", "X"])[codes]
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(letters))
    return path


def load_bank(cache_dir):
    """The visibility bank of the provided match, from the bootstrap cache."""
    z = np.load(os.path.join(cache_dir, "bootstrap.npz"))
    return degrade.visibility_bank(~z["missing"], z["dead"], blocks_of(z["period"]))
