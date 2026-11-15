import numpy as np

from possession import degrade


def test_stretch_keeps_both_ends_at_true_speed():
    mask = np.r_[np.ones(10, bool), np.zeros(80, bool), np.ones(5, bool)]      # visible head and tail
    for length in (40, 95, 400):
        out = degrade.stretch(mask, length)
        assert len(out) == length
        assert out[:10].all() and out[-5:].all() and not out[12:length - 7].any()


def test_bank_is_cut_along_segments():
    visible = np.r_[np.ones(50, bool), np.zeros(30, bool), np.ones(20, bool)]
    dead = np.r_[np.zeros(45, bool), np.ones(40, bool), np.zeros(15, bool)]
    bank = degrade.visibility_bank(visible, dead, [(1, 0, 100)])
    assert [len(m) for m in bank["dead"]] == [40] and [len(m) for m in bank["alive"]] == [45, 15]
    assert bank["dead"][0][:5].all() and not bank["dead"][0][5:35].any()


def test_degrade_is_deterministic_and_masks_only_the_ball(toy_match):
    bank = {"dead": [np.zeros(50, bool), np.r_[np.ones(10, bool), np.zeros(40, bool)]],
            "alive": [np.ones(150, bool), np.r_[np.ones(100, bool), np.zeros(20, bool), np.ones(30, bool)]]}
    a = degrade.degrade(toy_match, bank, seed=3)
    b = degrade.degrade(toy_match, bank, seed=3)
    c = degrade.degrade(toy_match, bank, seed=4)
    np.testing.assert_array_equal(a.ball_xy, b.ball_xy)
    np.testing.assert_array_equal(a.home_xy, b.home_xy)
    assert not np.array_equal(np.isnan(a.home_xy), np.isnan(c.home_xy))
    assert np.array_equal(a.label, toy_match.label)            # labels are never touched
    dead = toy_match.label == 2
    assert np.isnan(a.ball_xy[dead, 0]).mean() > 0.5 > np.isnan(a.ball_xy[~dead, 0]).mean()
