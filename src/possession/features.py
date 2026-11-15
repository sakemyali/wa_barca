"""Feature engineering shared by every tracking source.

One function, `build(match)`, turns a Match into three per-frame tables:

  players   collective movement of the two teams, no ball anywhere in it. Drives the alive/dead
            call, and is the only view that still works when the ball track is gone.
  ball      the ball against the players around it: distances, kinematics, touches, and how those
            evolve before and after the frame. Drives the team call.
  vis       where the frame sits in the ball track's visible / missing runs.

Conventions. Each half is processed on its own, so no window ever reaches across halftime. Inside a
half the pitch is rotated so the home team attacks towards +x, which makes a first half and a
second half look the same to a model. Slots are never assumed to be people: every player feature
is a statistic over the set of visible players of a team.

Features look both backwards and forwards in time. That is deliberate and allowed: the task is
offline labelling of a finished match, and the ground truth itself is retrospective (a possession
starts at the first touch of a sequence that only later proves to be controlled).
"""
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

from .schema import FPS

MAX_PLAYER_SPEED = 12.0      # m/s; anything faster is a tracker identity switch, not a sprint
TOUCH_DV = 3.5               # m/s change of ball velocity over 0.48 s that counts as a touch
TOUCH_WEAK_DV = 2.0          # m/s; weaker candidates, described to the model rather than decided here
TOUCH_RADIUS = 3.0           # m; a touch is credited to a team only if a player is this close
EXTRAP = 14                  # frames; how far outside a kick the straight-line motion is read
CLIP_TIME = 250              # frames; "time since / until" features saturate at 10 s


# ----------------------------------------------------------------------------------------------
# small vectorised helpers
# ----------------------------------------------------------------------------------------------

def roll_mean(f, lo, hi):
    """Mean of f over frames t+lo .. t+hi (inclusive), NaN-aware, clipped at the array ends."""
    n = len(f)
    ok = np.isfinite(f)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, f, 0.0), dtype=np.float64)))
    cn = np.concatenate(([0], np.cumsum(ok)))
    idx = np.arange(n)
    a = np.clip(idx + lo, 0, n)
    b = np.clip(idx + hi + 1, 0, n)
    cnt = cn[b] - cn[a]
    out = (cs[b] - cs[a]) / np.maximum(cnt, 1)
    out[cnt == 0] = np.nan
    return out.astype(np.float32)


def _shifted(g, shift, fill):
    """out[t] = g[t + shift], `fill` where that falls outside the array."""
    n = len(g)
    out = np.full(n, fill, g.dtype)
    if shift >= 0:
        out[:n - shift] = g[shift:] if shift else g
    else:
        out[-shift:] = g[:shift]
    return out


def _roll_extreme(f, lo, hi, minimum):
    """Min or max of f over frames t+lo .. t+hi, NaN-aware, clipped at the ends. hi - lo even."""
    fill = np.inf if minimum else -np.inf
    n = len(f)
    pad = max(abs(lo), abs(hi)) + 1
    g = np.concatenate([np.full(pad, fill), np.where(np.isfinite(f), f, fill), np.full(pad, fill)])
    filt = minimum_filter1d if minimum else maximum_filter1d
    h = filt(g, hi - lo + 1, mode="constant", cval=fill)
    out = h[np.arange(n) + pad + (lo + hi) // 2]
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def roll_min(f, lo, hi):
    return _roll_extreme(f, lo, hi, True)


def roll_max(f, lo, hi):
    return _roll_extreme(f, lo, hi, False)


def time_since(event):
    """Frames since `event` was last true (0 on the frame itself), CLIP_TIME if never / long ago."""
    idx = np.where(event, np.arange(len(event)), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(len(event)) - last, CLIP_TIME).astype(np.float32)


def time_until(event):
    return time_since(event[::-1])[::-1]


def central_velocity(xy, k):
    """(xy[t+k] - xy[t-k]) / (2k frames), in m/s; NaN where either end is missing."""
    v = np.full(xy.shape, np.nan, np.float32)
    if len(xy) > 2 * k:
        v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (FPS / (2.0 * k))
    return v


def nan_quantile_rows(a, qs):
    """Row-wise quantiles of a [n, k] array ignoring NaN; NaN for empty rows. Much faster than
    np.nanpercentile, which matters at 150k rows per match."""
    s = np.sort(a, axis=1)                       # NaN sorts last
    cnt = np.isfinite(a).sum(1)
    out = []
    rows = np.arange(len(a))
    for q in qs:
        pos = np.clip(np.round(q * (cnt - 1)).astype(np.int64), 0, a.shape[1] - 1)
        v = s[rows, pos]
        v[cnt == 0] = np.nan
        out.append(v)
    return out


def home_attacks_positive(home_xy, away_xy):
    """True if the home team defends the x = 0 goal during this block."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(home_xy[:, :, 0]) < np.nanmean(away_xy[:, :, 0])


def smooth_ball(ball_xy, sigma):
    """Gaussian low-pass of the ball track inside each visible run (never across a gap)."""
    if not sigma:
        return ball_xy
    out = ball_xy.copy()
    ok = np.isfinite(ball_xy[:, 0])
    edges = np.diff(np.concatenate(([0], ok.astype(np.int8), [0])))
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if e - s >= 3:
            out[s:e] = gaussian_filter1d(ball_xy[s:e].astype(np.float64), sigma, axis=0, mode="nearest")
    return out


# ----------------------------------------------------------------------------------------------
# players-only view
# ----------------------------------------------------------------------------------------------

def _team_stats(xy, vel, length, attack_positive):
    """Per-frame collective statistics of one team. xy [n, s, 2] already in the rotated frame;
    attack_positive says whether this team attacks +x, so depth is always 'metres from own goal'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y = xy[:, :, 0], xy[:, :, 1]
        depth = x if attack_positive else length - x
        fwd = vel[:, :, 0] if attack_positive else -vel[:, :, 0]
        speed = np.hypot(vel[:, :, 0], vel[:, :, 1])
        n = np.isfinite(x).sum(1).astype(np.float32)
        d10, d50, d90 = nan_quantile_rows(depth, (0.1, 0.5, 0.9))
        s50, s90 = nan_quantile_rows(speed, (0.5, 0.9))
        out = {
            "n": n,
            "depth_mean": np.nanmean(depth, 1), "depth_p10": d10, "depth_p90": d90,
            "depth_min": np.nanmin(depth, 1), "depth_max": np.nanmax(depth, 1),
            "y_mean": np.nanmean(y, 1), "y_std": np.nanstd(y, 1), "depth_std": np.nanstd(depth, 1),
            "sp_mean": np.nanmean(speed, 1), "sp_med": s50, "sp_p90": s90, "sp_max": np.nanmax(speed, 1),
            "slow": np.nanmean(np.where(np.isfinite(speed), speed < 1.0, np.nan), 1),
            "fast": np.nanmean(np.where(np.isfinite(speed), speed > 4.0, np.nan), 1),
            "fwd_mean": np.nanmean(fwd, 1),
            "vy_abs": np.nanmean(np.abs(vel[:, :, 1]), 1),
        }
    return {k: v.astype(np.float32) for k, v in out.items()}


def _nearest_opponent(a_xy, b_xy):
    """Mean over visible players of A of the distance to the closest player of B."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = np.empty(len(a_xy), np.float32)
        for s in range(0, len(a_xy), 20000):                      # chunked: the pairwise block is large
            a, b = a_xy[s:s + 20000], b_xy[s:s + 20000]
            d = np.sqrt(((a[:, :, None, :] - b[:, None, :, :]) ** 2).sum(-1))   # [n, sa, sb]
            d = np.where(np.isfinite(d), d, np.inf).min(2)
            d[~np.isfinite(d)] = np.nan
            out[s:s + 20000] = np.nanmean(d, 1)
        return out


def player_features(home_xy, away_xy, pitch):
    length, width = pitch
    vh = central_velocity(home_xy, 5)
    va = central_velocity(away_xy, 5)
    for v in (vh, va):
        v[np.hypot(v[:, :, 0], v[:, :, 1]) > MAX_PLAYER_SPEED] = np.nan
    th = _team_stats(home_xy, vh, length, True)
    ta = _team_stats(away_xy, va, length, False)
    f = {}
    for name, t in (("h", th), ("a", ta)):
        for k, v in t.items():
            f["{}_{}".format(name, k)] = v
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        both_speed = np.concatenate([np.hypot(vh[:, :, 0], vh[:, :, 1]), np.hypot(va[:, :, 0], va[:, :, 1])], 1)
        s25, s50, s75, s90 = nan_quantile_rows(both_speed, (0.25, 0.5, 0.75, 0.9))
        f["all_sp_mean"] = np.nanmean(both_speed, 1).astype(np.float32)
        f["all_sp_p25"], f["all_sp_med"], f["all_sp_p75"], f["all_sp_p90"] = s25, s50, s75, s90
        f["all_slow"] = np.nanmean(np.where(np.isfinite(both_speed), both_speed < 1.0, np.nan), 1).astype(np.float32)
        f["all_n"] = th["n"] + ta["n"]
        allxy = np.concatenate([home_xy, away_xy], 1)
        # how far the furthest player stands outside the lines (throw-in taker, corner taker)
        out_y = np.maximum(-allxy[:, :, 1], allxy[:, :, 1] - width)
        out_x = np.maximum(-allxy[:, :, 0], allxy[:, :, 0] - length)
        f["out_y_max"] = np.nanmax(out_y, 1).astype(np.float32)
        f["out_x_max"] = np.nanmax(out_x, 1).astype(np.float32)
        f["all_y_std"] = np.nanstd(allxy[:, :, 1], 1).astype(np.float32)
        f["all_x_std"] = np.nanstd(allxy[:, :, 0], 1).astype(np.float32)
        # acceleration of the group: change of the median speed over one second
    f["gap_depth"] = (length - ta["depth_mean"]) - th["depth_mean"]       # away centroid x - home centroid x
    f["gap_y"] = ta["y_mean"] - th["y_mean"]
    f["h_mark"] = _nearest_opponent(home_xy, away_xy)
    f["a_mark"] = _nearest_opponent(away_xy, home_xy)
    # symmetric combinations the alive/dead call can use without caring which team is which
    f["line_gap"] = (length - ta["depth_min"]) - th["depth_min"]          # distance between the two deepest players
    f["compact_x"] = th["depth_std"] + ta["depth_std"]

    base = ["all_sp_mean", "all_sp_med", "all_sp_p90", "all_slow", "h_sp_mean", "a_sp_mean",
            "h_fwd_mean", "a_fwd_mean", "h_depth_mean", "a_depth_mean", "all_n", "out_y_max"]
    ctx = {}
    for k in base:
        v = f[k]
        for w in (25, 75, 200):
            ctx["{}_past{}".format(k, w)] = roll_mean(v, -w, -1)
            ctx["{}_next{}".format(k, w)] = roll_mean(v, 1, w)
        ctx["{}_mid25".format(k)] = roll_mean(v, -12, 12)
    for k in ("all_sp_med", "all_sp_p90", "all_slow"):
        ctx[k + "_min150"] = roll_min(f[k], -75, 75)
        ctx[k + "_max150"] = roll_max(f[k], -75, 75)
        ctx[k + "_trend"] = ctx[k + "_next75"] - ctx[k + "_past75"]
    # collective drift of each team over the last / next 3 s, in its own attacking direction
    for name in ("h", "a"):
        d = f[name + "_depth_mean"]
        ctx[name + "_drift_past"] = d - _shifted(d, -75, np.nan)
        ctx[name + "_drift_next"] = _shifted(d, 75, np.nan) - d
    f.update(ctx)
    return f


# ----------------------------------------------------------------------------------------------
# ball view
# ----------------------------------------------------------------------------------------------

def _team_ball_distances(ball, xy, vel):
    """Nearest and second-nearest distance from ball to a team, the nearest player's relative
    speed to the ball, and head-counts inside 3 m and 8 m."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = np.sqrt(((xy - ball[:, None, :]) ** 2).sum(-1))
        dn = np.where(np.isfinite(d), d, np.inf)
        order = np.argsort(dn, axis=1)[:, :2]
        rows = np.arange(len(ball))
        d1 = dn[rows, order[:, 0]]
        d2 = dn[rows, order[:, 1]] if dn.shape[1] > 1 else np.full(len(ball), np.inf)
        v1 = vel[rows, order[:, 0]]
        c3 = (dn < 3.0).sum(1).astype(np.float32)
        c8 = (dn < 8.0).sum(1).astype(np.float32)
    bad = ~np.isfinite(ball[:, 0])
    for a in (d1, d2):
        a[~np.isfinite(a)] = np.nan
        a[bad] = np.nan
    c3[bad] = np.nan
    c8[bad] = np.nan
    return d1.astype(np.float32), d2.astype(np.float32), v1, c3, c8


def _nearest(point, xy):
    """Distance from a per-frame point [n, 2] to the closest visible player of xy [n, s, 2]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = np.sqrt(((xy - point[:, None, :]) ** 2).sum(-1))
        d = np.where(np.isfinite(d), d, np.inf).min(1)
    d[~np.isfinite(d)] = np.nan
    d[~np.isfinite(point[:, 0])] = np.nan
    return d.astype(np.float32)


def desmooth(ball_xy, vel):
    """Where the ball really was, undoing the tracker's smoothing around a kick.

    A low-pass filter rounds the corner of a trajectory at every touch, so at the instant of a
    kick the filtered ball is NOT at the kicker's feet -- it is cutting across the inside of the
    corner. But well before and well after the kick the filter has nothing to round, and the ball
    travels in a straight line. Extending the incoming line forwards and the outgoing line
    backwards to time t gives two estimates of the true position; at a kick they meet at the
    kick point. Returns (their mean [n, 2], the distance between them [n]).
    """
    n = len(ball_xy)
    k = EXTRAP
    fwd = np.full((n, 2), np.nan, np.float32)
    bwd = np.full((n, 2), np.nan, np.float32)
    if n > k:
        fwd[k:] = ball_xy[:-k] + vel[:-k] * (k / float(FPS))
        bwd[:-k] = ball_xy[k:] - vel[k:] * (k / float(FPS))
    both = np.isfinite(fwd[:, 0]) & np.isfinite(bwd[:, 0])
    q = np.where(both[:, None], (fwd + bwd) / 2, np.where(np.isfinite(fwd), fwd, bwd))
    q = np.where(np.isfinite(q), q, ball_xy).astype(np.float32)
    gap = np.where(both, np.hypot(fwd[:, 0] - bwd[:, 0], fwd[:, 1] - bwd[:, 1]), np.nan).astype(np.float32)
    return q, gap


def _chain(n, times, attrs, offsets):
    """Describe, for every frame, the touches around it.

    times: sorted frame indices of the touch candidates. attrs: {name: value per candidate}.
    offsets: which neighbours to describe, 0 = the last one at or before the frame, -1 the one
    before that, 1 the next one, 2 the one after. Returns {"<name>_<offset>": [n]} plus
    "dt_<offset>" (frames from the frame to that touch, clipped).
    """
    out = {}
    idx = np.searchsorted(times, np.arange(n), side="right") - 1          # last touch at or before t
    for off in offsets:
        j = idx + off
        ok = (j >= 0) & (j < len(times))
        jj = np.clip(j, 0, max(len(times) - 1, 0))
        tag = "m{}".format(-off) if off <= 0 else "p{}".format(off)
        if len(times):
            dt = np.where(ok, np.abs(times[jj] - np.arange(n)), CLIP_TIME)
        else:
            dt = np.full(n, CLIP_TIME)
        out["tc_dt_" + tag] = np.minimum(dt, CLIP_TIME).astype(np.float32)
        for name, vals in attrs.items():
            v = np.where(ok, vals[jj], np.nan) if len(times) else np.full(n, np.nan)
            out["tc_{}_{}".format(name, tag)] = v.astype(np.float32)
    return out


def ball_features(ball_xy, home_xy, away_xy, pitch):
    length, width = pitch
    n = len(ball_xy)
    vis = np.isfinite(ball_xy[:, 0])
    f = {}
    bx, by = ball_xy[:, 0], ball_xy[:, 1]
    f["bx"] = bx.astype(np.float32)
    f["by_edge"] = np.minimum(by, width - by).astype(np.float32)          # metres inside the nearer touchline
    f["bx_edge"] = np.minimum(bx, length - bx).astype(np.float32)         # metres inside the nearer goal line
    f["b_out"] = np.where(vis, np.maximum(-np.minimum(f["by_edge"], f["bx_edge"]), 0.0), np.nan).astype(np.float32)

    v2 = central_velocity(ball_xy, 2)
    v6 = central_velocity(ball_xy, 6)
    sp2 = np.hypot(v2[:, 0], v2[:, 1])
    sp6 = np.hypot(v6[:, 0], v6[:, 1])
    f["b_sp"] = sp2
    f["b_sp6"] = sp6
    f["b_vx"] = v2[:, 0]                                                   # + = towards the away goal
    f["b_vy_abs"] = np.abs(v2[:, 1])
    # |change of velocity| over half a second. The provided ball track is low-pass filtered, so a
    # kick is a ramp of that length rather than a step; a shorter baseline would miss most of it.
    dv = np.full(n, np.nan, np.float32)
    dv[6:-6] = np.hypot(v2[12:, 0] - v2[:-12, 0], v2[12:, 1] - v2[:-12, 1])
    f["b_dv"] = dv
    dsp = np.full(n, np.nan, np.float32)
    dsp[6:-6] = sp2[12:] - sp2[:-12]                                       # speeding up (kick) or slowing down
    f["b_dsp"] = dsp

    vh = central_velocity(home_xy, 5)
    va = central_velocity(away_xy, 5)
    dH, dH2, vH, cH3, cH8 = _team_ball_distances(ball_xy, home_xy, vh)
    dA, dA2, vA, cA3, cA8 = _team_ball_distances(ball_xy, away_xy, va)
    f["dH"], f["dA"], f["dH2"], f["dA2"] = dH, dA, dH2, dA2
    f["d_diff"] = dH - dA                                                  # < 0: home is closer
    f["d_min"] = np.fmin(dH, dA)
    f["cH3"], f["cA3"], f["cH8"], f["cA8"] = cH3, cA3, cH8, cA8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f["relH"] = np.hypot(v2[:, 0] - vH[:, 0], v2[:, 1] - vH[:, 1])     # ball moving with the nearest home player?
        f["relA"] = np.hypot(v2[:, 0] - vA[:, 0], v2[:, 1] - vA[:, 1])
    # soft control: 1 when a player is on the ball, fading over a couple of metres
    ctrlH = np.exp(-dH / 1.5)
    ctrlA = np.exp(-dA / 1.5)
    edge = ctrlH - ctrlA
    f["ctrl_edge"] = edge.astype(np.float32)
    closer = np.where(np.isfinite(f["d_diff"]), (f["d_diff"] < 0).astype(np.float32), np.nan)

    for w in (6, 12, 25, 50, 100, 200):
        f["edge_past{}".format(w)] = roll_mean(edge, -w, -1)
        f["edge_next{}".format(w)] = roll_mean(edge, 1, w)
        f["closer_past{}".format(w)] = roll_mean(closer, -w, -1)
        f["closer_next{}".format(w)] = roll_mean(closer, 1, w)
    for w in (12, 50):
        f["dH_minpast{}".format(w)] = roll_min(dH, -w, 0)
        f["dA_minpast{}".format(w)] = roll_min(dA, -w, 0)
        f["dH_minnext{}".format(w)] = roll_min(dH, 0, w)
        f["dA_minnext{}".format(w)] = roll_min(dA, 0, w)
        f["sp_past{}".format(w)] = roll_mean(sp2, -w, -1)
        f["sp_next{}".format(w)] = roll_mean(sp2, 1, w)
    f["dv_maxpast12"] = roll_max(dv, -12, 0)
    f["dv_maxnext12"] = roll_max(dv, 0, 12)

    # touches: a sharp change of ball velocity with a player close by. The team of the last touch
    # is the mechanical definition of "who owns the ball", so these are the core team features.
    peak = np.zeros(n, bool)
    dvf = np.where(np.isfinite(dv), dv, 0.0)
    peak[1:-1] = (dvf[1:-1] >= TOUCH_DV) & (dvf[1:-1] >= dvf[:-2]) & (dvf[1:-1] > dvf[2:])
    near = np.fmin(dH, dA) < TOUCH_RADIUS
    touchH = peak & near & (dH <= dA)
    touchA = peak & near & (dA < dH)
    f["tH_since"], f["tA_since"] = time_since(touchH), time_since(touchA)
    f["tH_until"], f["tA_until"] = time_until(touchH), time_until(touchA)
    f["touch_last"] = np.sign(f["tA_since"] - f["tH_since"])               # +1 home touched last
    f["touch_next"] = np.sign(f["tA_until"] - f["tH_until"])
    for w in (75, 200):
        f["touchH_past{}".format(w)] = roll_mean(touchH.astype(np.float32), -w, -1) * w
        f["touchA_past{}".format(w)] = roll_mean(touchA.astype(np.float32), -w, -1) * w
        f["touchH_next{}".format(w)] = roll_mean(touchH.astype(np.float32), 1, w) * w
        f["touchA_next{}".format(w)] = roll_mean(touchA.astype(np.float32), 1, w) * w
    # tight contact: a player within 1.2 m, the nearest thing to "has the ball at his feet"
    onH = dH < 1.2
    onA = dA < 1.2
    f["onH_since"], f["onA_since"] = time_since(onH), time_since(onA)
    f["onH_until"], f["onA_until"] = time_until(onH), time_until(onA)
    f["on_last"] = np.sign(f["onA_since"] - f["onH_since"])
    f["on_next"] = np.sign(f["onA_until"] - f["onH_until"])

    # the same proximity questions asked of the de-smoothed ball position
    q, gap = desmooth(ball_xy, v2)
    qH, qA = _nearest(q, home_xy), _nearest(q, away_xy)
    f["q_gap"] = gap
    f["qH"], f["qA"] = qH, qA
    f["q_diff"] = qH - qA
    qedge = np.exp(-qH / 1.5) - np.exp(-qA / 1.5)
    f["q_edge"] = qedge.astype(np.float32)
    for w in (12, 25, 50, 100):
        f["qedge_past{}".format(w)] = roll_mean(qedge, -w, -1)
        f["qedge_next{}".format(w)] = roll_mean(qedge, 1, w)

    # touch chain: every local peak of velocity change, weak ones included, described by who was
    # nearest to the de-smoothed kick point and by how hard the ball was redirected
    weak = np.zeros(n, bool)
    weak[1:-1] = (dvf[1:-1] >= TOUCH_WEAK_DV) & (dvf[1:-1] >= dvf[:-2]) & (dvf[1:-1] > dvf[2:])
    times = np.flatnonzero(weak)
    margin = (qA - qH)[times]                      # > 0: a home player was nearer the kick point
    attrs = {"margin": margin, "dmin": np.fmin(qH, qA)[times], "dv": dv[times]}
    f.update(_chain(n, times, attrs, (-1, 0, 1, 2)))
    return f


def visibility_features(ball_xy):
    vis = np.isfinite(ball_xy[:, 0])
    f = {"miss": (~vis).astype(np.float32),
         "seen_since": time_since(vis), "seen_until": time_until(vis),
         "miss_since": time_since(~vis), "miss_until": time_until(~vis)}
    f["miss_frac_past75"] = roll_mean(f["miss"], -75, -1)
    f["miss_frac_next75"] = roll_mean(f["miss"], 1, 75)
    return f


# ----------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------

def rotate(xy, pitch):
    """180-degree rotation of the pitch: what 'the teams changed ends' does to coordinates."""
    return np.asarray(pitch, np.float32) - xy


def build(match, ball_sigma=0.0, views=("players", "ball", "vis")):
    """Feature tables for every frame of both halves.

    Returns (names, X [n_rows, n_features] float32, rows [n_rows] frame indices into the match,
    view_of [n_features] the view each column belongs to).
    """
    tables, rows = [], []
    for _, start, stop in match.blocks():
        home, away, ball = match.home_xy[start:stop], match.away_xy[start:stop], match.ball_xy[start:stop]
        if not home_attacks_positive(home, away):
            home, away, ball = rotate(home, match.pitch), rotate(away, match.pitch), rotate(ball, match.pitch)
        ball = smooth_ball(ball, ball_sigma)
        f = {}
        if "players" in views:
            f.update(("p:" + k, v) for k, v in player_features(home, away, match.pitch).items())
        if "ball" in views:
            f.update(("b:" + k, v) for k, v in ball_features(ball, home, away, match.pitch).items())
        if "vis" in views:
            f.update(("v:" + k, v) for k, v in visibility_features(ball).items())
        tables.append(f)
        rows.append(np.arange(start, stop))
    names = list(tables[0])
    X = np.concatenate([np.stack([t[k] for k in names], 1) for t in tables]).astype(np.float32)
    return names, X, np.concatenate(rows)


def swap_teams(match):
    """The same match seen from the other bench: home and away exchanged, H and A exchanged.

    Used to double the training data and, more importantly, to make every model exactly
    indifferent to which side is called 'home'.
    """
    label = match.label.copy()
    label[match.label == 0] = 1
    label[match.label == 1] = 0
    return match._replace(name=match.name + "_swap", home_xy=match.away_xy, away_xy=match.home_xy, label=label)
