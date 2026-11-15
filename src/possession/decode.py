"""Turn per-frame probabilities into a sequence of possessions.

A per-frame argmax flickers, and the ranking metric counts possessions, not frames: every flicker
is an invented possession and usually destroys a real one. The decoder is a three-state hidden
Markov model (H, A, D) solved exactly with Viterbi. It has no minimum duration and no smoothing
window; a short possession survives whenever the evidence for it outweighs the price of entering
and leaving it, and the prices differ by kind of transition because the events differ: a change
of team inside live play is common and can be brief, while a stoppage is rarer and never lasts a
quarter of a second.
"""
import numpy as np

from .schema import A, D, H

EPS = 1e-6


class DecoderParams(tuple):
    """(temper, cost_team, cost_stop, cost_restart, gain_team, gain_stop, gain_restart).

    Emissions are log-probabilities times `temper` (consecutive frames are far from independent,
    so their evidence is discounted). Each cost is paid once per switch of that kind: H<->A,
    alive->D, D->alive. Each gain scales how much an event detector may move its cost up or down
    on a given frame (0 = ignore the detector): a change of team is cheap on the frame where a
    touch by the other team was detected and dearer everywhere else.
    """
    __slots__ = ()

    def __new__(cls, temper=0.3, cost_team=6.0, cost_stop=6.0, cost_restart=6.0,
                gain_team=0.0, gain_stop=0.0, gain_restart=0.0):
        return super().__new__(cls, tuple(float(v) for v in (
            temper, cost_team, cost_stop, cost_restart, gain_team, gain_stop, gain_restart)))

    def __getnewargs__(self):                  # so that pickling (joblib workers) round-trips
        return tuple(self)

    temper = property(lambda self: self[0])
    cost_team = property(lambda self: self[1])
    cost_stop = property(lambda self: self[2])
    cost_restart = property(lambda self: self[3])
    gain_team = property(lambda self: self[4])
    gain_stop = property(lambda self: self[5])
    gain_restart = property(lambda self: self[6])


def event_scores(p_event, floor=1e-4):
    """Log-ratio of an event detector's probability to its own average, per frame.

    Zero where the detector says nothing unusual, positive on a detected event, negative where it
    is confident nothing happens. p_event [n] or [n, k].
    """
    p = np.clip(np.asarray(p_event, np.float64), floor, 1.0)
    return np.log(p) - np.log(np.clip(p.mean(0), floor, 1.0))


def emissions(p_dead, p_home):
    """Log-probability of each state per frame, [n, 3] in the order H, A, D."""
    p_dead = np.clip(p_dead, EPS, 1 - EPS)
    p_home = np.clip(p_home, EPS, 1 - EPS)
    alive = np.log1p(-p_dead)
    return np.stack([alive + np.log(p_home), alive + np.log1p(-p_home), np.log(p_dead)], 1)


# which entry of (team, stop, restart) each transition belongs to; -1 on the diagonal
_KIND = np.full((3, 3), -1)
_KIND[H, A] = _KIND[A, H] = 0
_KIND[H, D] = _KIND[A, D] = 1
_KIND[D, H] = _KIND[D, A] = 2
_MASK = np.stack([(_KIND == k).astype(np.float64) for k in range(3)])       # [kind, from, to]


def viterbi(log_emit, params, events=None):
    """Most probable state path. log_emit [n, 3] (H, A, D); events [n, 3] event_scores for
    (team change, stop, restart) or None. Returns int8 [n] of label codes."""
    return viterbi_batch(log_emit, [params], events)[0]


def viterbi_batch(log_emit, params_list, events=None):
    """Viterbi for many parameter sets at once (same emissions). Returns int8 [len(params), n].

    A transition into frame t is priced with the event scores of frame t. Decoding the same
    matches under a few hundred settings is what tuning the decoder means, hence the batching.
    """
    P = np.array([tuple(p) for p in params_list], np.float64)                # [B, 7]
    B, n = len(P), len(log_emit)
    trans = -np.einsum("bk,kij->bij", P[:, 1:4], _MASK)                      # [B, from, to]
    use_events = events is not None and bool(np.any(P[:, 4:7]))
    if use_events:
        gain = np.einsum("bk,kij->bkij", P[:, 4:7], _MASK)                   # [B, kind, from, to]
    temper = P[:, 0][:, None]
    score = log_emit[0][None, :] * temper                                    # [B, 3]
    back = np.zeros((n, B, 3), np.int8)
    rows = np.arange(B)[:, None]
    cols = np.arange(3)[None, :]
    for t in range(1, n):
        cand = score[:, :, None] + trans
        if use_events:
            cand = cand + np.einsum("bkij,k->bij", gain, events[t])
        best = cand.argmax(1)                                                # [B, to]
        back[t] = best
        score = cand[rows, best, cols] + log_emit[t][None, :] * temper
    path = np.empty((B, n), np.int8)
    path[:, -1] = score.argmax(1)
    rows = np.arange(B)
    for t in range(n - 1, 0, -1):
        path[:, t - 1] = back[t, rows, path[:, t]]
    return path


def decode(p_dead, p_home, blocks, params, n_frames=None, fill=3, p_events=None):
    """Decode each half independently; frames outside every block get `fill` (X by default).
    p_events [n, 3]: detector probabilities of (team change, stop, restart), or None."""
    return decode_batch(p_dead, p_home, blocks, [params], n_frames, fill, p_events)[0]


def decode_batch(p_dead, p_home, blocks, params_list, n_frames=None, fill=3, p_events=None):
    n = n_frames if n_frames is not None else len(p_dead)
    out = np.full((len(params_list), n), fill, np.int8)
    for _, start, stop in blocks:
        ev = None if p_events is None else event_scores(p_events[start:stop])
        out[:, start:stop] = viterbi_batch(emissions(p_dead[start:stop], p_home[start:stop]), params_list, ev)
    return out


def forward_backward(log_emit, log_trans, log_start):
    """Posterior state probabilities of a small HMM, [n, k]. Scaled recursions, exact."""
    n, k = log_emit.shape
    emit = np.exp(log_emit - log_emit.max(1, keepdims=True))
    trans = np.exp(log_trans)
    alpha = np.empty((n, k))
    a = np.exp(log_start) * emit[0]
    alpha[0] = a / a.sum()
    for t in range(1, n):
        a = (alpha[t - 1] @ trans) * emit[t]
        alpha[t] = a / a.sum()
    beta = np.ones(k)
    post = np.empty((n, k))
    post[-1] = alpha[-1]
    for t in range(n - 2, -1, -1):
        beta = trans @ (emit[t + 1] * beta)
        beta /= beta.sum()
        g = alpha[t] * beta
        post[t] = g / g.sum()
    return post


def majority_smooth(labels, blocks, window):
    """The organisers' baseline smoother (centred majority vote), kept as a comparison point."""
    out = labels.copy()
    for _, start, stop in blocks:
        seg = labels[start:stop]
        counts = np.stack([np.concatenate(([0], np.cumsum(seg == k))) for k in (H, A, D)], 1)
        idx = np.arange(len(seg))
        lo = np.maximum(0, idx - window)
        hi = np.minimum(len(seg), idx + window + 1)
        out[start:stop] = (counts[hi] - counts[lo]).argmax(1)
    return out
