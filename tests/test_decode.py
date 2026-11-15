import numpy as np

from possession import decode as dec


def _emit(rng, n):
    return np.log(rng.dirichlet(np.ones(3), n))


def test_batch_viterbi_equals_single():
    rng = np.random.default_rng(1)
    e = _emit(rng, 400)
    params = [dec.DecoderParams(0.3, 2, 4, 6), dec.DecoderParams(1.0, 0.5, 0.5, 0.5), dec.DecoderParams(0.1, 9, 1, 3)]
    batch = dec.viterbi_batch(e, params)
    for i, p in enumerate(params):
        assert np.array_equal(batch[i], dec.viterbi(e, p))


def test_viterbi_is_optimal_on_a_small_case():
    rng = np.random.default_rng(2)
    e = _emit(rng, 7)
    p = dec.DecoderParams(1.0, 0.7, 1.1, 0.4)
    cost = {(0, 1): 0.7, (1, 0): 0.7, (0, 2): 1.1, (1, 2): 1.1, (2, 0): 0.4, (2, 1): 0.4}
    best, best_path = -1e18, None
    for code in range(3 ** 7):
        path = [(code // 3 ** k) % 3 for k in range(7)]
        s = sum(e[t, path[t]] for t in range(7)) - sum(cost.get((path[t - 1], path[t]), 0.0) for t in range(1, 7))
        if s > best:
            best, best_path = s, path
    assert dec.viterbi(e, p).tolist() == best_path


def test_costly_switches_remove_flicker():
    p_dead = np.full(300, 0.01)
    p_home = np.full(300, 0.9)
    p_home[100:103] = 0.2                       # a three-frame blip of weak evidence
    out = dec.decode(p_dead, p_home, [(1, 0, 300)], dec.DecoderParams(0.3, 6, 6, 6))
    assert (out == 0).all()
    p_home[100:160] = 0.02                      # sixty frames of strong evidence is a possession
    out = dec.decode(p_dead, p_home, [(1, 0, 300)], dec.DecoderParams(0.3, 6, 6, 6))
    assert (out[105:155] == 1).all() and (out[:95] == 0).all()
