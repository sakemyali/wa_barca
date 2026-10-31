import os

import numpy as np
import pytest

from possession.homography import load_homography, project, unproject


def test_round_trip():
    H = np.array([[1000., 50, 20], [30, 600, 10], [0.01, 0.02, 1]])
    x, y = np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.5, 0.8])
    u, v = project(H, x, y)
    assert u.shape == (3,)
    x2, y2 = unproject(H, u, v)
    np.testing.assert_allclose([x2, y2], [x, y], atol=1e-9)


def test_identity_and_nan():
    assert project(np.eye(3), 0.25, 0.75) == (0.25, 0.75)
    assert np.isnan(project(np.full((3, 3), np.nan), 0.5, 0.5)[0])


@pytest.mark.skipif(not os.environ.get("MTAH_DATA_DIR"), reason="MTAH_DATA_DIR unset")
def test_real_file():
    H, err, valid = load_homography()
    assert H.shape == (199319, 3, 3) and err.shape == valid.shape == (199319,)
    assert valid.all()
