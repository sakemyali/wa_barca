import numpy as np

from possession import kickoff


def _teams(n, cross_at):
    """Two lines of five players each side of halfway; three home players cross at `cross_at`."""
    t = np.arange(n)[:, None]
    y = np.linspace(10, 58, 5)[None, :] + 0 * t
    home_x = np.full((n, 5), 40.0)
    away_x = np.full((n, 5), 65.0)
    home_x[cross_at:, :3] = 60.0
    return (np.stack([home_x, y], 2).astype(np.float32), np.stack([away_x, y], 2).astype(np.float32))


def test_half_that_starts_at_the_whistle_needs_no_correction():
    home, away = _teams(2000, cross_at=30)
    assert kickoff.opening_kickoff(home, away, (105.0, 68.0)) == 0


def test_kickoff_is_found_after_a_long_wait_in_formation():
    home, away = _teams(4000, cross_at=1500)
    k = kickoff.opening_kickoff(home, away, (105.0, 68.0))
    assert k == 1500 - kickoff.CROSSING_DELAY


def test_people_walking_to_their_places_are_not_a_kickoff():
    home, away = _teams(6000, cross_at=3000)
    home[:800, :, 0] = 70.0                     # the home team starts on the wrong side of the pitch
    k = kickoff.opening_kickoff(home, away, (105.0, 68.0))
    assert k == 3000 - kickoff.CROSSING_DELAY


def test_orientation_does_not_matter():
    home, away = _teams(4000, cross_at=1500)
    flip = np.array([105.0, 68.0], np.float32)
    assert kickoff.opening_kickoff(flip - home, flip - away, (105.0, 68.0)) == 1500 - kickoff.CROSSING_DELAY
