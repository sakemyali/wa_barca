"""Make a clean public match look like the provided automatic tracking.

Public tracking is corrected and complete; the provided tracking is neither. A model trained on
the first and run on the second would lean on signals that are simply not there. So before any
training, every public match is degraded in the three ways that were measured on the provided one
(none of which needs a label):

  ball smoothing   the provided ball track is heavily low-pass filtered: a kick shows up as an
                   acceleration pulse about 0.8 s wide, against 0.1 s in raw optical tracking.
                   Public balls are filtered until their pulses, their velocity changes and
                   their distance to the nearest player have the same distributions.
  ball position    the provided ball sits a little further from the nearest player than a
                   corrected ball does. A slow random offset is added to match; it has to be
                   larger for Metrica's sample games, whose ball is snapped onto the player at
                   every event.
  ball visibility  the provided ball track comes and goes in long runs, mostly but not only with
                   stoppages. The pattern is copied, not modelled: visibility masks are cut out of
                   the provided match along its own estimated stoppages and live spells, and
                   pasted onto public stoppages and live spells of similar length.
  players          a random couple of tracks are blanked for random stretches, so that no model
                   learns to expect twenty-two complete trajectories.

During a stoppage the public sources have no usable ball (DFL freezes it, Metrica blanks it), so
where the pasted mask says "visible" the ball is drawn on a straight line from where play stopped
to where it restarts. That is a placeholder, and the models are kept away from fine ball kinematics
when judging alive/dead for exactly that reason (see models.py).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .features import smooth_ball
from .schema import D

# Calibrated without labels against the provided match (experiments/calibration.md): Gaussian
# sigma in frames, and the standard deviation in metres of a slowly varying positional offset.
BALL_SIGMA = 7.0
BALL_NOISE = {"dfl": 0.25, "metrica": 0.80}
NOISE_CORRELATION = 12.0  # frames


def runs(mask):
    """Half-open (start, stop) of every run of True."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def visibility_bank(visible, dead, blocks):
    """Visibility masks of the provided match, cut along its (estimated) dead and alive segments.

    visible, dead: bool [n]. blocks: [(period, start, stop)]. Returns {"dead": [...], "alive": [...]}.
    """
    bank = {"dead": [], "alive": []}
    for _, start, stop in blocks:
        d = dead[start:stop]
        v = visible[start:stop]
        for key, mask in (("dead", d), ("alive", ~d)):
            for s, e in runs(mask):
                bank[key].append(v[s:e].copy())
    return bank


def stretch(mask, length):
    """Resize a visibility mask to `length` frames, keeping what happens at its two ends at true
    speed (how long the ball stays visible after play stops, how early it reappears) and
    stretching only the middle."""
    n = len(mask)
    if n == length:
        return mask.copy()
    head = min(n, length) // 2
    out = np.empty(length, bool)
    out[:head] = mask[:head]
    out[length - head:] = mask[n - head:]
    mid = length - 2 * head
    if mid > 0:
        src = mask[head:n - head]
        if len(src) == 0:
            src = mask[max(head - 1, 0):max(head, 1)]
        out[head:length - head] = src[(np.arange(mid) * len(src) // mid)]
    return out


def sample_mask(masks, length, rng, k=8):
    """A mask for a segment of `length` frames: one of the k bank entries closest in duration."""
    lens = np.array([len(m) for m in masks], np.float64)
    order = np.argsort(np.abs(np.log(lens) - np.log(max(length, 1))), kind="stable")[:k]
    return stretch(masks[int(rng.choice(order))], length)


def synthetic_dead_ball(ball_xy, dead):
    """Ball track with every stoppage replaced by a straight line between the last live position
    before it and the first live position after it."""
    out = ball_xy.copy()
    out[dead] = np.nan
    ok = np.isfinite(out[:, 0])
    if not ok.any():
        return out
    t = np.arange(len(out))
    for c in (0, 1):
        out[~ok, c] = np.interp(t[~ok], t[ok], out[ok, c])
    return out


def drop_players(xy, rng, rate=1.0 / 1500, mean_len=150):
    """Blank random stretches of random tracks. `rate` is the chance per frame and track that a
    blank stretch starts; lengths are geometric with mean `mean_len` frames."""
    out = xy.copy()
    n, slots = xy.shape[:2]
    for s in range(slots):
        starts = np.flatnonzero(rng.random(n) < rate)
        for a in starts:
            out[a:a + 1 + rng.geometric(1.0 / mean_len), s] = np.nan
    return out


def ball_noise(n, sd, rng):
    """A smooth random 2-D offset of standard deviation `sd` metres."""
    if not sd:
        return np.zeros((n, 2), np.float32)
    noise = gaussian_filter1d(rng.standard_normal((n, 2)), NOISE_CORRELATION, axis=0, mode="nearest")
    return (noise * (sd / noise.std())).astype(np.float32)


def degrade(match, bank, seed):
    """An ATD-like copy of a labelled public match. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    sd = next((v for k, v in BALL_NOISE.items() if match.name.startswith(k)), 0.0)
    ball = match.ball_xy.astype(np.float32).copy()
    visible = np.ones(match.n_frames, bool)
    for _, start, stop in match.blocks():
        dead = match.label[start:stop] == D
        seg_ball = smooth_ball(synthetic_dead_ball(ball[start:stop], dead), BALL_SIGMA)
        ball[start:stop] = seg_ball + ball_noise(stop - start, sd, rng)
        for key, mask in (("dead", dead), ("alive", ~dead)):
            for s, e in runs(mask):
                visible[start + s:start + e] = sample_mask(bank[key], e - s, rng)
    ball[~visible] = np.nan
    return match._replace(ball_xy=ball, home_xy=drop_players(match.home_xy, rng),
                          away_xy=drop_players(match.away_xy, rng))
