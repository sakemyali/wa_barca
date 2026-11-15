import numpy as np

from possession import features


def test_roll_mean_and_min_match_brute_force():
    rng = np.random.default_rng(0)
    f = rng.normal(size=200).astype(np.float32)
    f[[3, 50, 51, 120]] = np.nan
    for lo, hi in ((-5, -1), (1, 8), (-6, 6), (0, 12)):
        ref_mean, ref_min, ref_max = [], [], []
        for t in range(len(f)):
            w = f[max(t + lo, 0):max(t + hi + 1, 0)]
            w = w[np.isfinite(w)]
            ref_mean.append(w.mean() if len(w) else np.nan)
            ref_min.append(w.min() if len(w) else np.nan)
            ref_max.append(w.max() if len(w) else np.nan)
        np.testing.assert_allclose(features.roll_mean(f, lo, hi), ref_mean, rtol=1e-5, atol=1e-6)
        if (hi - lo) % 2 == 0:
            np.testing.assert_allclose(features.roll_min(f, lo, hi), ref_min, rtol=1e-6)
            np.testing.assert_allclose(features.roll_max(f, lo, hi), ref_max, rtol=1e-6)


def test_time_since_and_until():
    e = np.zeros(20, bool)
    e[[4, 9]] = True
    since, until = features.time_since(e), features.time_until(e)
    assert since[4] == 0 and since[6] == 2 and since[9] == 0 and since[12] == 3
    assert since[0] == features.CLIP_TIME                      # nothing has happened yet
    assert until[0] == 4 and until[5] == 4 and until[10] == features.CLIP_TIME


def test_no_feature_looks_across_halftime(toy_match):
    names, X, rows = features.build(toy_match)
    changed = toy_match._replace(ball_xy=toy_match.ball_xy.copy(), home_xy=toy_match.home_xy.copy())
    changed.ball_xy[800:] += 3.0                               # disturb the second half only
    changed.home_xy[800:] += 1.0
    _, X2, _ = features.build(changed)
    first = rows < 800
    np.testing.assert_array_equal(X[first], X2[first])


def test_second_half_is_rotated_to_look_like_the_first(toy_match):
    names, X, rows = features.build(toy_match, views=("players",))
    col = names.index("p:gap_depth")                           # away centroid x minus home centroid x
    assert np.nanmean(X[rows < 800, col]) > 0 and np.nanmean(X[rows >= 800, col]) > 0


def test_swapping_teams_mirrors_the_team_features(toy_match):
    names, X, _ = features.build(toy_match)
    _, Xs, _ = features.build(features.swap_teams(toy_match))
    for a, b, sign in (("b:dH", "b:dA", 1), ("p:h_sp_mean", "p:a_sp_mean", 1), ("b:d_diff", "b:d_diff", -1)):
        np.testing.assert_allclose(X[:, names.index(a)], sign * Xs[:, names.index(b)], rtol=1e-4, atol=1e-3)


def test_desmoothing_recovers_the_kick_point():
    # a ball that travels +x at 10 m/s, is kicked at frame 100, and leaves along +y at 10 m/s
    n, kick = 200, 100
    t = np.arange(n)
    true = np.stack([np.where(t < kick, 0.4 * t, 0.4 * kick), np.where(t < kick, 0.0, 0.4 * (t - kick))], 1)
    smooth = features.smooth_ball(true.astype(np.float32), 7.0)
    q, gap = features.desmooth(smooth, features.central_velocity(smooth, 2))
    corner = true[kick]
    assert np.linalg.norm(smooth[kick] - corner) > 1.0          # the filter cut the corner
    assert np.linalg.norm(q[kick] - corner) < 0.5 * np.linalg.norm(smooth[kick] - corner)
    assert gap[40] < 0.05                                       # nothing to undo on a straight line
